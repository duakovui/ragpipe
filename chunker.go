// Package chunker cắt một tài liệu Markdown/text dài thành các chunk ~N token,
// dùng cho pipeline RAG (embedding + vector store).
//
// Nguyên tắc thiết kế:
//  1. Cắt theo ranh giới TỰ NHIÊN: xuống dòng hoặc dấu kết câu (. ! ? … ;)
//     -> không bao giờ cắt ngang giữa 1 câu/1 dòng.
//  2. Trước khi đếm token & tạo embedding, mọi ảnh (markdown/html/url ảnh trần)
//     bị xóa sạch -> đỡ tốn token vô ích vì ảnh không mang ngữ nghĩa cho LLM.
//  3. Mỗi chunk vẫn giữ StartOffset/EndOffset trỏ đúng vào tài liệu GỐC (đầy đủ
//     ảnh) -> map ngược 1-1 để hiển thị/trích dẫn lại bản gốc khi cần, dù
//     embedding chỉ được tính trên bản đã xóa ảnh.
package ragpipe

import (
	"regexp"
	"strings"
)

// ---------- Token counter (pluggable) ----------

// TokenCounterFunc đếm số token của 1 đoạn text.
//
// Khuyến nghị: bind vào tiktoken-go để đếm đúng theo tokenizer OpenAI
// (go get github.com/pkoukk/tiktoken-go):
//
//	enc, _ := tiktoken.GetEncoding("o200k_base") // dùng cho gpt-4o-mini / gpt-4.1
//	counter := func(s string) int { return len(enc.Encode(s, nil, nil)) }
//	chunks := chunker.SplitToChunks(doc, counter, chunker.Options{})
//
// Nếu truyền nil, package dùng ApproxTokenCounter (ước lượng, không cần thêm dependency).
type TokenCounterFunc func(s string) int

// ApproxTokenCounter ước lượng nhanh số token khi chưa cần độ chính xác tuyệt đối.
// Heuristic: ~3 rune/token (mức trung bình an toàn cho cl100k/o200k với văn bản
// tiếng Việt có dấu lẫn tiếng Anh).
func ApproxTokenCounter(s string) int {
	n := len([]rune(strings.TrimSpace(s)))
	if n == 0 {
		return 0
	}
	t := n / 3
	if t == 0 {
		t = 1
	}
	return t
}

// ---------- Xóa ảnh ----------

var (
	reMarkdownImg = regexp.MustCompile(`!\[[^\]]*\]\([^)]*\)`)
	reHTMLImg     = regexp.MustCompile(`(?i)<img[^>]*/?>`)
	reBareImgURL  = regexp.MustCompile(`(?i)https?://\S+\.(?:png|jpe?g|gif|webp|svg)\S*`)
	reBlankLines  = regexp.MustCompile(`\n{3,}`)
	reMdEndMarker = regexp.MustCompile(`<!--\s*THE END\s*-->`)
)

// stripImages xóa mọi dạng tham chiếu ảnh khỏi text, dọn lại dòng trống dư ra.
func stripImages(s string) string {
	s = reMarkdownImg.ReplaceAllString(s, "")
	s = reHTMLImg.ReplaceAllString(s, "")
	s = reBareImgURL.ReplaceAllString(s, "")
	s = reMdEndMarker.ReplaceAllString(s, "")
	s = reBlankLines.ReplaceAllString(s, "\n\n")
	return s
}

// ---------- Segment: đơn vị nhỏ nhất, không bao giờ bị cắt đôi ----------

type segment struct {
	Start, End int
	Raw        string // nguyên văn trong tài liệu gốc (có thể chứa ảnh)
	Clean      string // đã xóa ảnh, dùng để đếm token & tạo embedding
	Tokens     int
}

// reBoundary khớp 1 ranh giới cắt hợp lệ: 1+ dấu xuống dòng, HOẶC dấu kết câu
// (. ! ? …  ;) có thể theo sau bởi dấu đóng ngoặc/ngoặc kép rồi đến khoảng trắng.
var reBoundary = regexp.MustCompile(`\n+|[.!?…;]+["'’”)\]]*[ \t]+`)

// splitSegments cắt doc thành các segment liền kề mà ghép lại đúng bằng doc gốc
// (đảm bảo không mất/lệch ký tự nào -> map ngược offset luôn chính xác).
func splitSegments(doc string) []segment {
	idxs := reBoundary.FindAllStringIndex(doc, -1)
	segs := make([]segment, 0, len(idxs)+1)
	start := 0
	for _, m := range idxs {
		end := m[1]
		if end <= start {
			continue
		}
		segs = append(segs, segment{Start: start, End: end, Raw: doc[start:end]})
		start = end
	}
	if start < len(doc) {
		segs = append(segs, segment{Start: start, End: len(doc), Raw: doc[start:]})
	}
	return segs
}

// ---------- Chunk: kết quả cuối cùng, đẩy vào vector store ----------

type Chunk struct {
	Index       int
	RawText     string // nguyên văn (CÓ ảnh) - dùng để map ngược / hiển thị lại bản gốc
	CleanText   string // đã xóa ảnh - dùng để tạo embedding / feed vào prompt LLM
	StartOffset int    // vị trí bắt đầu trong doc gốc (byte offset)
	EndOffset   int    // vị trí kết thúc trong doc gốc (byte offset)
	TokenCount  int    // số token của CleanText
}

// ---------- Options ----------

type Options struct {
	TargetTokens  int // mục tiêu số token/chunk, mặc định 500
	MaxTokens     int // ngưỡng cứng trước khi buộc phải chốt chunk, mặc định TargetTokens*1.2
	OverlapTokens int // số token overlap giữa 2 chunk liên tiếp, mặc định 0 (tắt)
}

func (o *Options) setDefaults() {
	if o.TargetTokens <= 0 {
		o.TargetTokens = 500
	}
	if o.MaxTokens <= 0 {
		o.MaxTokens = o.TargetTokens + o.TargetTokens/5 // +20%
	}
	if o.OverlapTokens < 0 {
		o.OverlapTokens = 0
	}
}

// ---------- Thuật toán chính ----------

// SplitToChunks cắt doc thành các Chunk ~Options.TargetTokens token (đã xóa ảnh),
// không bao giờ cắt ngang câu/dòng. Mỗi Chunk vẫn giữ Start/EndOffset để map
// ngược về bản gốc (có ảnh) khi cần lưu/hiển thị trong vector store.
func SplitToChunks(doc string, counter TokenCounterFunc, opt Options) []Chunk {
	if counter == nil {
		counter = ApproxTokenCounter
	}
	opt.setDefaults()

	segs := splitSegments(doc)
	for i := range segs {
		segs[i].Clean = stripImages(segs[i].Raw)
		segs[i].Tokens = counter(segs[i].Clean)
	}

	var (
		chunks []Chunk
		cur    []segment
		curTok int
	)

	flush := func() {
		if len(cur) == 0 {
			return
		}
		start := cur[0].Start
		end := cur[len(cur)-1].End

		var cleanB strings.Builder
		for _, s := range cur {
			cleanB.WriteString(s.Clean)
		}
		cleanText := strings.TrimSpace(cleanB.String())

		chunks = append(chunks, Chunk{
			Index:       len(chunks),
			RawText:     doc[start:end],
			CleanText:   cleanText,
			StartOffset: start,
			EndOffset:   end,
			TokenCount:  counter(cleanText),
		})

		if opt.OverlapTokens > 0 {
			// giữ lại 1 phần đuôi của chunk vừa chốt làm phần đầu của chunk kế tiếp
			var keep []segment
			keepTok := 0
			for i := len(cur) - 1; i >= 0; i-- {
				if keepTok+cur[i].Tokens > opt.OverlapTokens {
					break
				}
				keep = append([]segment{cur[i]}, keep...)
				keepTok += cur[i].Tokens
			}
			cur, curTok = keep, keepTok
		} else {
			cur, curTok = nil, 0
		}
	}

	for _, s := range segs {
		// segment rỗng sau khi xóa ảnh (vd: dòng chỉ chứa ảnh/marker) -> gộp thẳng,
		// không tính vào ngưỡng token, không kích hoạt flush
		if strings.TrimSpace(s.Clean) == "" {
			cur = append(cur, s)
			continue
		}

		// thêm segment này sẽ vượt MaxTokens -> chốt chunk hiện tại trước
		if curTok > 0 && curTok+s.Tokens > opt.MaxTokens {
			flush()
		}

		cur = append(cur, s)
		curTok += s.Tokens

		// đã đạt mục tiêu -> chốt chunk (segment tự nó dài hơn MaxTokens, hiếm
		// gặp, vẫn được nhận trọn vẹn vì nguyên tắc #1 là không cắt ngang câu)
		if curTok >= opt.TargetTokens {
			flush()
		}
	}
	flush()

	return chunks
}
