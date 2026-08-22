package ragpipe

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	readability "codeberg.org/readeck/go-readability/v2"
	html2md "github.com/JohannesKaufmann/html-to-markdown/v2"
	"github.com/PuerkitoBio/goquery"
	"golang.org/x/net/html"
)

type CrawlResult struct {
	Url      string
	Title    string
	Markdown string
}

type Crawler struct {
	numWorker  int
	bufferSize int
	client     *http.Client
}

type crawlEvent struct {
	url    string
	result *CrawlResult
	err    *CustomError
	done   bool
}

type CrawlConfig struct {
	GetArticle         func(doc *goquery.Document, url string) (*Article, error)
	IgnoreContent      func(doc *goquery.Document, url string) bool
	IgnoreLink		   func(url string) bool	
	LinkSelector       func(doc *goquery.Document) *goquery.Selection
}

type Article struct {
	Title   string
	Content *html.Node
}

func defaultGetArtical(doc *goquery.Document, baseUrl string) (*Article, error) {
	pageURL, _ := url.Parse(baseUrl)
	article, err := readability.FromDocument(doc.Selection.Get(0), pageURL)
	if err != nil {
		return nil, err
	}
	return &Article{
		Title:   article.Title(),
		Content: article.Node,
	}, nil
}

func NewCrawler(numWorker, bufferSize int, timeout time.Duration) *Crawler {
	if numWorker < 1 {
		numWorker = 1
	}
	if bufferSize < 1 {
		bufferSize = 1
	}

	return &Crawler{
		numWorker:  numWorker,
		bufferSize: bufferSize,
		client:     &http.Client{Timeout: timeout},
	}
}

func (c *Crawler) CrawlPage(ctx context.Context, baseUrl string, getArticle func(doc *goquery.Document, url string) (*Article, error)) (string, []byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseUrl, nil)
	if err != nil {
		return "", nil, err
	}

	res, err := c.client.Do(req)
	if err != nil {
		return "", nil, err
	}

	if res.StatusCode != http.StatusOK {
		statusCode := res.StatusCode
		res.Body.Close()
		return "", nil, fmt.Errorf("http %d", statusCode)
	}

	doc, err := goquery.NewDocumentFromReader(res.Body)
	res.Body.Close()
	if err != nil {
		return "", nil, err
	}

	var article *Article
	if getArticle != nil {
		article, err = getArticle(doc, baseUrl)
	} else {
		article, err = defaultGetArtical(doc, baseUrl)
	}
	if err != nil {
		return "", nil, err
	}

	md, err := html2md.ConvertNode(article.Content)
	if err != nil {
		return "", nil, err
	}

	return article.Title, md, nil
}

func (c *Crawler) CrawlRecursive(ctx context.Context, baseUrl string, config *CrawlConfig, errCh chan<- CustomError) (chan CrawlResult, *atomic.Int64) {
	counter := new(atomic.Int64)
	results := make(chan CrawlResult, c.bufferSize)
	urls := make(chan string, c.numWorker)
	events := make(chan crawlEvent, c.numWorker)
	crawledLinks := make(map[string]struct{})
	var crawledLinksMu sync.Mutex

	markCrawled := func(link string) bool {
		crawledLinksMu.Lock()
		defer crawledLinksMu.Unlock()

		if _, exists := crawledLinks[link]; exists {
			return false
		}
		crawledLinks[link] = struct{}{}
		return true
	}
	queue := make([]string, 0)

	if canonical, ok := normalizeCrawlURL(baseUrl, baseUrl); ok {
		baseUrl = canonical
		queue = append(queue, baseUrl)
	} else {
		reportErr(errCh, CustomError{
			Stage: StageCrawl,
			Op:    "init",
			URL:   baseUrl,
			Err:   "invalid start link",
		})
	}

	var cfg CrawlConfig
	if config != nil {
		cfg = *config
	}

	// init default func
	if cfg.GetArticle == nil {
		cfg.GetArticle = defaultGetArtical
	}

	var workerWG sync.WaitGroup
	workerWG.Add(c.numWorker)
	for range c.numWorker {
		go c.worker(ctx, urls, events, cfg, &workerWG, markCrawled, counter)
	}

	go func() {
		closeAndWait := func() {
			close(urls)
			workerWG.Wait()
			close(results)
		}
		inFlight := 0

		for {
			if ctx.Err() != nil || (len(queue) == 0 && inFlight == 0) {
				closeAndWait()
				return
			}

			var nextURL chan string
			var next string
			if len(queue) > 0 && inFlight < c.numWorker {
				nextURL = urls
				next = queue[0]
			}

			select {
			case <-ctx.Done():
				closeAndWait()
				return
			case event := <-events:
				if event.url != "" {
					queue = append(queue, event.url)
				}
				if event.result != nil {
					select {
					case results <- *event.result:
					case <-ctx.Done():
					}
				}
				if event.err != nil {
					reportErr(errCh, *event.err)
				}
				if event.done {
					inFlight--
				}
			case nextURL <- next:
				queue = queue[1:]
				inFlight++
			}
		}
	}()

	return results, counter
}

func normalizeCrawlURL(baseURL, rawURL string) (string, bool) {
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return "", false
	}

	base, err := url.Parse(baseURL)
	if err != nil || !base.IsAbs() || base.Host == "" {
		return "", false
	}

	ref, err := url.Parse(rawURL)
	if err != nil {
		return "", false
	}

	if !ref.IsAbs() {
		ref = base.ResolveReference(ref)
	}

	if (ref.Scheme != "http" && ref.Scheme != "https") || ref.Host == "" {
		return "", false
	}

	if !strings.EqualFold(ref.Host, base.Host) {
		return "", false
	}

	ref.Fragment = ""
	ref.Path = strings.TrimRight(ref.Path, "/")
	
	return ref.String(), true
}

func (c *Crawler) worker(ctx context.Context, urls <-chan string, events chan<- crawlEvent, config CrawlConfig, workerWG *sync.WaitGroup, markCrawled func(string) bool, counter *atomic.Int64) {
	defer workerWG.Done()

	sendResult := func(res CrawlResult) {
		select {
		case events <- crawlEvent{result: &res}:
		case <-ctx.Done():
		}
	}
	sendDone := func() {
		select {
		case events <- crawlEvent{done: true}:
		case <-ctx.Done():
		}
	}
	sendJob := func(baseURL string, url string) {
		canonical, ok := normalizeCrawlURL(baseURL, url)
		if !ok {
			return
		}
		if config.IgnoreLink != nil && config.IgnoreLink(canonical) {
			return
		}
		if !markCrawled(canonical) {
			return
		}
		url = canonical
		select {
		case events <- crawlEvent{url: url}:
		case <-ctx.Done():
		}
	}
	fail := func(op string, url string, err error) {
		select {
		case events <- crawlEvent{err: &CustomError{
			Stage: StageCrawl,
			Op:    op,
			URL:   url,
			Err:   err.Error(),
		}}:
		case <-ctx.Done():
		}
	}

	processJob := func(baseUrl string) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseUrl, nil)
		if err != nil {
			fail("new_request", baseUrl, err)
			return
		}

		res, err := c.client.Do(req)
		if err != nil {
			fail("http_get", baseUrl, err)
			return
		}

		if res.StatusCode != http.StatusOK {
			fail("http_status", baseUrl, fmt.Errorf("HTTP %d", res.StatusCode))
			res.Body.Close()
			return
		}

		doc, err := goquery.NewDocumentFromReader(res.Body)
		res.Body.Close()
		if err != nil {
			fail("parse_html", baseUrl, err)
			return
		}

		if config.LinkSelector != nil {
			config.LinkSelector(doc).Each(func(i int, a *goquery.Selection) {
				sendJob(baseUrl, a.AttrOr("href", ""))
			})
		} else {
			doc.Find("a").Each(func(i int, a *goquery.Selection) {
				sendJob(baseUrl, a.AttrOr("href", ""))
			})
		}

		if config.IgnoreContent != nil && config.IgnoreContent(doc, baseUrl) {
			return
		}

		article, err := config.GetArticle(doc, baseUrl)
		if err != nil {
			fail("parse_article", baseUrl, err)
			return
		}
		if article.Title == "" {
			fail("parse_article", baseUrl, fmt.Errorf("missing title"))
			return
		}

		md, err := html2md.ConvertNode(article.Content)
		if err != nil {
			fail("parse_markdown", baseUrl, err)
			return
		}
		sendResult(CrawlResult{Url: baseUrl, Title: article.Title, Markdown: string(md)})
	}

	for {
		select {
		case <-ctx.Done():
			return
		case url, ok := <-urls:
			if !ok {
				return
			}
			counter.Add(1)
			processJob(url)
			sendDone()
		}
	}
}
