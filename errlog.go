// Package main: custom error pipeline.
//
// Mọi lỗi phát sinh trong crawl -> embed -> upsert đều được gói thành 1
// CustomError rồi đẩy vào 1 channel chung (errCh). main chạy 1 goroutine
// (StartErrorLogger) đọc hết channel đó và ghi xuống file errors.log dưới
// dạng JSON Lines (mỗi dòng 1 object) -> dễ grep/jq. Với embedding, payload
// retry được tách thành file riêng trong logs/embedding để errors.log chỉ còn
// metadata và UUID tham chiếu.
package ragpipe

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ErrorStage cho biết lỗi xảy ra ở giai đoạn nào của pipeline.
type ErrorStage string

const (
	StageCrawl  ErrorStage = "crawl"  // HTTP request, parse HTML, html2md
	StageEmbed  ErrorStage = "embed"  // gọi API embedding, marshal payload
	StageVector ErrorStage = "vector" // upsert vào vector store (qdrant)
)

// CustomError là bản ghi lỗi duy nhất chạy suốt pipeline.
// Các trường chi tiết giúp log giàu thông tin. Jobs được giữ trong bộ nhớ tới
// lúc logger ghi file retry, nhưng không được ghi trực tiếp vào errors.log.
//
// Cách dùng điển hình trong worker:
//
//	reportErr(errCh, CustomError{
//	    Stage: StageEmbed,
//	    Op:    "create_embedding",
//	    Jobs:  batch,        // nguyên batch lỗi -> logger lưu để retry sau
//	    Err:   err.Error(),
//	})
type CustomError struct {
	Timestamp time.Time      `json:"ts"`                   // thời điểm xảy ra
	Stage     ErrorStage     `json:"stage"`                // crawl / embed / vector
	Op        string         `json:"op"`                   // thao tác cụ thể: http_get, parse_html, create_embedding, json_marshal, upsert...
	URL       string         `json:"url,omitempty"`        // link đang xử lý
	Document  string         `json:"doc,omitempty"`        // tên tài liệu (CrawlJob.Name)
	BatchSize int            `json:"batch_size,omitempty"` // số chunk trong batch (embed/vector)
	BatchUUID string         `json:"batch_uuid,omitempty"` // file retry trong logs/embedding
	Jobs      []EmbeddingJob `json:"-"`                    // payload được logger lưu thành file riêng
	Err       string         `json:"error"`                // message lỗi gốc
}

// Error implement error interface (dành cho các path có cần error thuần).
func (e *CustomError) Error() string {
	return fmt.Sprintf("[%s] op=%s url=%s doc=%q batch=%d: %s",
		e.Stage, e.Op, e.URL, e.Document, e.BatchSize, e.Err)
}

// reportErr gửi 1 CustomError vào errCh. Blocking send: nếu channel đầy, worker
// chịu đợi (back-pressure) -> bảo đảm KHÔNG mất lỗi nào. Logger liên tục drain
// nên không có khả năng deadlock (logger chỉ dừng khi closeFn được gọi ở cuối main).
func reportErr(errCh chan<- CustomError, ce CustomError) {
	if ce.Timestamp.IsZero() {
		ce.Timestamp = time.Now()
	}
	errCh <- ce
}

// StartErrorLogger mở file log và chạy 1 goroutine đọc mọi CustomError từ
// channel trả về, ghi dạng JSON Lines. Với lỗi có Jobs, logger tạo UUID và
// ghi mảng jobs vào {uuid}.json trong logs/embedding. Trả về:
//   - errCh: channel để các worker push lỗi vào
//   - closeFn: gọi (thường defer ở main) sau khi toàn bộ pipeline xong để
//     flush nốt các lỗi còn nằm trong buffer rồi đóng file.
//
// Buffer 4096: đủ lớn để các worker hiếm khi bị block ngay cả khi ổ đĩo chậm.
func StartErrorLogger(path string) (errCh chan CustomError, closeFn func() error, err error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, nil, err
	}

	embeddingDir := embeddingJobsDir(path)
	if err := os.MkdirAll(embeddingDir, 0o755); err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("create embedding log directory: %w", err)
	}

	errCh = make(chan CustomError, 4096)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		enc := json.NewEncoder(f)
		enc.SetEscapeHTML(false)
		for ce := range errCh {
			if len(ce.Jobs) > 0 {
				batchUUID, saveErr := saveEmbeddingJobs(embeddingDir, ce.Jobs)
				ce.BatchUUID = batchUUID
				if saveErr != nil {
					ce.Err = fmt.Sprintf("%s (save embedding jobs: %v)", ce.Err, saveErr)
				}
				ce.Jobs = nil
			}
			if e := enc.Encode(ce); e != nil {
				// fallback: nếu JSON encode lỗi (hiếm), vẫn ghi thô để không mất log.
				fmt.Fprintf(f, "ENCODE_FAIL: stage=%s op=%s err=%s\n", ce.Stage, ce.Op, ce.Err)
			}
		}
	}()

	closeFn = func() error {
		close(errCh)
		wg.Wait()
		return f.Close()
	}
	return errCh, closeFn, nil
}

func embeddingJobsDir(errorLogPath string) string {
	logDir := filepath.Dir(errorLogPath)
	if filepath.Base(logDir) == "logs" {
		return filepath.Join(logDir, "embedding")
	}
	return filepath.Join(logDir, "logs", "embedding")
}

func saveEmbeddingJobs(dir string, jobs []EmbeddingJob) (string, error) {
	batchUUID := uuid.NewString()
	b, err := json.MarshalIndent(jobs, "", "  ")
	if err != nil {
		return batchUUID, err
	}
	b = append(b, '\n')

	path := filepath.Join(dir, batchUUID+".json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return batchUUID, err
	}
	return batchUUID, nil
}
