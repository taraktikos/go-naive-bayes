package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
)

const (
	positive = "positive"
	negative = "negative"
)

/* Example:
{
	word: "restaurant",
	counter: {
		positive: 2,
		negative: 0,
	}
}
 */
type wordFrequency struct {
	word    string
	counter map[string]int
}

/*	Example:
{
	dataset: {
		positive: [
			"The restaurant is excellent",
			"Second sentence"
		],
		negative: [
			"Some negative
		]
	},
	words: {
		restaurant: {
			word: "restaurant",
			counter: {
				positive: 2,
				negative: 0,
			}
		}
	}
}
*/
type classifier struct {
	dataset map[string][]string
	words   map[string]wordFrequency
}

func main() {
	nb := newClassifier()
	dataset := dataset("./datasets/imdb_labelled.txt")
	nb.train(dataset)

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("Enter your text: ")
		sentence, _ := reader.ReadString('\n')
		result := nb.classify(sentence)
		class := negative
		if result[positive] > result[negative] {
			class = positive
		}

		fmt.Printf("> Your text is %s\n\n", class)
	}
}

func newClassifier() *classifier {
	return &classifier{
		dataset: map[string][]string{},
		words:   map[string]wordFrequency{},
	}
}

func dataset(file string) map[string]string {
	f, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	dataset := make(map[string]string)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		l := scanner.Text()
		data := strings.Split(l, "\t")
		if len(data) != 2 {
			continue
		}
		sentence := data[0]
		switch data[1] {
		case "1":
			dataset[sentence] = positive
		default:
			dataset[sentence] = negative
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	return dataset
}

// train populates a classifier's dataset and words with input dataset map
// Sample dataset: map[string]string{
//	"The restaurant is excellent": "Positive",
//	"I really love this restaurant": "Positive",
//	"Their food is awful": "Negative",
//}
func (c *classifier) train(dataset map[string]string) {
	for sentence, class := range dataset {
		c.addSentence(sentence, class)
		words := tokenize(sentence)
		for _, w := range words {
			c.addWord(w, class)
		}
	}
}

// classify return the probabilities of a sentence being each class
// Sample @return map[string]float64 {
//	"positive": 0.7,
//	"negative": 0.1,
//}
// Meaning 70% chance the input sentence is positive, 10% it's negative
func (c *classifier) classify(sentence string) map[string]float64 {
	words := tokenize(sentence)
	posProb := c.probability(words, positive)
	negProb := c.probability(words, negative)
	return map[string]float64{
		positive: posProb,
		negative: negProb,
	}
}

func (c *classifier) addSentence(sentence, class string) {
	c.dataset[class] = append(c.dataset[class], sentence)
}

func (c *classifier) addWord(word, class string) {
	wf, ok := c.words[word]
	if !ok {
		wf = wordFrequency{
			word:    word,
			counter: map[string]int{positive: 0, negative: 0},
		}
	}
	wf.counter[class] ++
	c.words[word] = wf
}

func (c *classifier) priorProb(class string) float64 {
	classCount := float64(len(c.dataset[class]))
	totalCount := float64(len(c.dataset[positive]) + len(c.dataset[negative]))
	return classCount / totalCount
}

func (c *classifier) wordCount(class string) int {
	count := 0
	for _, wf := range c.words {
		count += wf.counter[positive]
	}
	return count
}

func (c *classifier) totalWordCount() int {
	return c.wordCount(positive) + c.wordCount(negative)
}

func (c *classifier) totalDistinctWordCount() int {
	posCount := 0
	negCount := 0
	for _, wf := range c.words {
		posCount += zeroOneTransform(wf.counter[positive])
		negCount += zeroOneTransform(wf.counter[negative])
	}
	return posCount + negCount
}

// https://medium.com/@kcatstack/sentiment-analysis-naive-bayes-classifier-from-scratch-part-1-theory-4949115ba13
func (c *classifier) probability(words []string, class string) float64 {
	prob := c.priorProb(class)
	for _, w := range words {
		count := 0
		if wf, ok := c.words[w]; ok {
			count = wf.counter[class]
		}
		prob *= float64(count+1) / float64(c.wordCount(class)+c.totalDistinctWordCount())
	}
	for _, w := range words {
		count := 0
		if wf, ok := c.words[w]; ok {
			count = wf.counter[positive] + wf.counter[negative]
		}
		prob /= float64(count+1) / float64(c.totalWordCount()+c.totalDistinctWordCount())
	}

	return prob
}

func zeroOneTransform(x int) int {
	//return int(math.Ceil(float64(x) / (float64(x) + 1)))
	if x == 0 {
		return 0
	}
	return 1
}
