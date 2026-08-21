package ragpipe

import (
	"context"
	"encoding/json"
	"sync"
)

type EmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type EmbeddingResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

type EmbeddingClient interface {
	CreateBatch(ctx context.Context, input []string) (EmbeddingResponse, error)
}

type Payload struct {
	Index    int    `json:"index"`
	Document string `json:"document"`
	Link     string `json:"link"`
	Content  string `json:"content"`
}

type EmbeddingJob struct {
	CleanText string  `json:"clean_text"`
	Payload   Payload `json:"payload"`
}

type EmbeddingResult struct {
	Vector  []float32
	Payload json.RawMessage
}

type EmbeddingProcessor struct {
	numWorker   int
	batchSize   int
	countToken  TokenCounterFunc
	chunkOption Options
	client    EmbeddingClient
	vectorStore VectorStore
	wg          sync.WaitGroup
}

type VectorStore interface {
	Upsert(ctx context.Context, batch []EmbeddingResult) error
}

func NewEmbeddingProcessor(numWorker, batchSize int, countToken TokenCounterFunc, chunkOption Options, client EmbeddingClient, vectorStore VectorStore) *EmbeddingProcessor {
	return &EmbeddingProcessor{
		numWorker:   numWorker,
		batchSize:   batchSize,
		countToken:  countToken,
		chunkOption: chunkOption,
		client: client,
		vectorStore: vectorStore,
	}
}

func (e *EmbeddingProcessor) Process(ctx context.Context, results <-chan CrawlResult, errCh chan<- CustomError) {
	batchChan := make(chan []EmbeddingJob, 100)
	batch := make([]EmbeddingJob, 0, e.batchSize)

	for range e.numWorker {
		e.wg.Add(1)
		go e.worker(ctx, batchChan, errCh)
	}

	sendBatch := func(batch []EmbeddingJob) {
		select {
		case <-ctx.Done():
		case batchChan <- batch:
		}
	}

	for res := range results {
		if ctx.Err() != nil {
			break
		}
		chunks := SplitToChunks(res.Markdown, e.countToken, e.chunkOption)
		for _, chunk := range chunks {
			batch = append(batch, EmbeddingJob{
				CleanText: chunk.CleanText,
				Payload: Payload{
					Index:    chunk.Index,
					Document: res.Title,
					Link:     res.Url,
					Content:  chunk.RawText,
				},
			})
			if len(batch) == e.batchSize {
				sendBatch(batch)
				batch = make([]EmbeddingJob, 0, e.batchSize)
			}
		}
	}
	if len(batch) > 0 && ctx.Err() == nil {
		sendBatch(batch)
	}

	close(batchChan)
	e.wg.Wait()
}

// worker xử lý từng batch: embed -> marshal payload -> upsert. Ở mỗi bước lỗi,
// đẩy CustomError vào errCh kèm đủ thông tin để retry:
//   - create_embedding lỗi: nguyên batch (vì chưa embed được gì).
//   - json_marshal lỗi: chỉ đúng 1 job bị lỗi.
//   - upsert lỗi: chỉ các job thực sự đã embed+marshal OK (đó là cái cần re-upsert).
func (e *EmbeddingProcessor) worker(ctx context.Context, batchs <-chan []EmbeddingJob, errCh chan<- CustomError) {
	for batch := range batchs {
		input := make([]string, len(batch))
		for i, chunk := range batch {
			input[i] = chunk.CleanText
		}

		result, err := e.client.CreateBatch(ctx, input)
		if err != nil {
			reportErr(errCh, CustomError{
				Stage:     StageEmbed,
				Op:        "create_embedding",
				BatchSize: len(batch),
				Jobs:      batch,
				Err:       err.Error(),
			})
			continue
		}

		embeddingResults := make([]EmbeddingResult, 0, len(result.Data))
		succeeded := make([]EmbeddingJob, 0, len(result.Data))
		for _, data := range result.Data {
			job := batch[data.Index]
			payload, err := json.Marshal(job.Payload)
			if err != nil {
				reportErr(errCh, CustomError{
					Stage:     StageEmbed,
					Op:        "json_marshal",
					URL:       job.Payload.Link,
					Document:  job.Payload.Document,
					BatchSize: 1,
					Jobs:      []EmbeddingJob{job},
					Err:       err.Error(),
				})
				continue
			}
			embeddingResults = append(embeddingResults, EmbeddingResult{
				Vector:  data.Embedding,
				Payload: payload,
			})
			succeeded = append(succeeded, job)
		}

		if len(embeddingResults) == 0 {
			continue
		}

		if err := e.vectorStore.Upsert(ctx, embeddingResults); err != nil {
			reportErr(errCh, CustomError{
				Stage:     StageVector,
				Op:        "upsert",
				BatchSize: len(embeddingResults),
				Jobs:      succeeded,
				Err:       err.Error(),
			})
		}
	}
	e.wg.Done()
}
