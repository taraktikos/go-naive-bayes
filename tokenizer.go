package main

import (
	"regexp"
	"strings"
)

// stopWords are words which have very little meaning
var stopWords = map[string]struct{}{
	"i": {}, "me": {}, "my": {}, "myself": {}, "we": {}, "our": {}, "ours": {},
	"ourselves": {}, "you": {}, "your": {}, "yours": {}, "yourself": {}, "yourselves": {},
	"he": {}, "him": {}, "his": {}, "himself": {}, "she": {}, "her": {}, "hers": {},
	"herself": {}, "it": {}, "its": {}, "itself": {}, "they": {}, "them": {}, "their": {},
	"theirs": {}, "themselves": {}, "what": {}, "which": {}, "who": {}, "whom": {}, "this": {},
	"that": {}, "these": {}, "those": {}, "am": {}, "is": {}, "are": {}, "was": {},
	"were": {}, "be": {}, "been": {}, "being": {}, "have": {}, "has": {}, "had": {},
	"having": {}, "do": {}, "does": {}, "did": {}, "doing": {}, "a": {}, "an": {},
	"the": {}, "and": {}, "but": {}, "if": {}, "or": {}, "because": {}, "as": {},
	"until": {}, "while": {}, "of": {}, "at": {}, "by": {}, "for": {}, "with": {},
	"about": {}, "against": {}, "between": {}, "into": {}, "through": {}, "during": {},
	"before": {}, "after": {}, "above": {}, "below": {}, "to": {}, "from": {}, "up": {},
	"down": {}, "in": {}, "out": {}, "on": {}, "off": {}, "over": {}, "under": {},
	"again": {}, "further": {}, "then": {}, "once": {}, "here": {}, "there": {}, "when": {},
	"where": {}, "why": {}, "how": {}, "all": {}, "any": {}, "both": {}, "each": {},
	"few": {}, "more": {}, "most": {}, "other": {}, "some": {}, "such": {}, "no": {},
	"nor": {}, "not": {}, "only": {}, "same": {}, "so": {}, "than": {}, "too": {},
	"very": {}, "can": {}, "will": {}, "just": {}, "don't": {}, "should": {}, "should've": {},
	"now": {}, "aren't": {}, "couldn't": {}, "didn't": {}, "doesn't": {}, "hasn't": {}, "haven't": {},
	"isn't": {}, "shouldn't": {}, "wasn't": {}, "weren't": {}, "won't": {}, "wouldn't": {},
}

func isStopWord(w string) bool {
	_, ok := stopWords[w]
	return ok
}

func clenup(sentence string) string {
	re := regexp.MustCompile("[^a-zA-Z 0-9]+")
	return re.ReplaceAllLiteralString(strings.ToLower(sentence), "")
}

func tokenize(sentence string) []string {
	s := clenup(sentence)
	words := strings.Fields(s)
	var tokens []string
	for _, w := range words {
		if !isStopWord(w) {
			tokens = append(tokens, w)
		}
	}
	return tokens
}
