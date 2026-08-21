package ragpipe_test

import (
	"strings"
	"testing"
	"github.com/duakovui/ragpipe"
)

// ---------------------------------------------------------------
// 1. SplitToChunks - unit test thuần, không cần dịch vụ ngoài.
// ---------------------------------------------------------------

// counter đơn giản: 1 token ~ 4 ký tự, đủ để ép chunker chia nhỏ văn bản.
func fakeCounter(s string) int {
	n := len(strings.TrimSpace(s)) / 4
	if n == 0 && strings.TrimSpace(s) != "" {
		n = 1
	}
	return n
}

func TestSplitToChunks(t *testing.T) {
	// Văn bản gồm nhiều câu, đủ dài để phải chia thành nhiều chunk.
	doc := strings.Repeat("Đây là một câu kiểm tra cho chunker. ", 200)

	opt := ragpipe.Options{TargetTokens: 50, OverlapTokens: 10}
	chunks := ragpipe.SplitToChunks(doc, fakeCounter, opt)

	if len(chunks) < 2 {
		t.Fatalf("mong đợi ít nhất 2 chunk, nhận được %d", len(chunks))
	}

	for i, c := range chunks {
		// Index đánh liên tục từ 0.
		if c.Index != i {
			t.Errorf("chunk %d có Index=%d", i, c.Index)
		}
		// CleanText không được rỗng.
		if strings.TrimSpace(c.CleanText) == "" {
			t.Errorf("chunk %d có CleanText rỗng", i)
			continue
		}
		// Không chunk nào (trừ chunk cuối) vượt quá MaxTokens.
		maxTok := opt.TargetTokens + opt.TargetTokens/5
		if i < len(chunks)-1 && c.TokenCount > maxTok {
			t.Errorf("chunk %d có %d token > MaxTokens=%d", i, c.TokenCount, maxTok)
		}
		// Offset phải khớp với slice trong doc gốc.
		if c.EndOffset <= c.StartOffset {
			t.Errorf("chunk %d có offset sai: [%d, %d)", i, c.StartOffset, c.EndOffset)
		}
		if c.EndOffset > len(doc) {
			t.Errorf("chunk %d vượt quá độ dài doc", i)
		}
		if got := doc[c.StartOffset:c.EndOffset]; got != c.RawText {
			t.Errorf("chunk %d: RawText không khớp slice doc", i)
		}
		// Ghép toàn bộ RawText phải khớp doc (không mất/lệch ký tự).
	}

	var rebuilt strings.Builder
	for _, c := range chunks {
		rebuilt.WriteString(c.RawText)
	}
	if rebuilt.String() != doc {
		t.Errorf("ghép các chunk lại không ra doc gốc (có overlap -> trùng lặp)")
	}
}