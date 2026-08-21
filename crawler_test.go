package ragpipe_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/duakovui/ragpipe"
)

func TestCrawlRecursiveWaitsForInFlightJobs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		if r.URL.Path == "/" {
			fmt.Fprint(w, `<html><body>
<a href="/one">one</a>
<a href="/two">two</a>
</body></html>`)
			return
		}
		fmt.Fprintf(w, "<html><body><p>%s</p></body></html>", r.URL.Path)
	}))
	defer server.Close()

	crawler := ragpipe.NewCrawler(1, 1, time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	results, counter := crawler.CrawlRecursive(ctx, server.URL+"/", &ragpipe.CrawlConfig{
		GetArticle: func(doc *goquery.Document, pageURL *url.URL) (*ragpipe.Article, error) {
			return &ragpipe.Article{
				Title:   pageURL.Path,
				Content: doc.Find("body").Get(0),
			}, nil
		},
	}, make(chan ragpipe.CustomError, 16))

	collected := make(chan []ragpipe.CrawlResult, 1)
	go func() {
		var got []ragpipe.CrawlResult
		for result := range results {
			got = append(got, result)
		}
		collected <- got
	}()

	var got []ragpipe.CrawlResult
	select {
	case got = <-collected:
	case <-time.After(2 * time.Second):
		cancel()
		t.Fatal("CrawlRecursive did not finish after processing the initial page")
	}

	if counter.Load() != 3 {
		t.Fatalf("expected 3 crawled pages, got %d", counter.Load())
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 results, got %d", len(got))
	}
}
