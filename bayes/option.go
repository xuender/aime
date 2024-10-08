package bayes

import (
	"cmp"

	"github.com/xuender/aime/prob"
)

type Option[C, V cmp.Ordered] func(*Classifier[C, V])

func OptionLaplaceSmoothing[C, V cmp.Ordered](classifier *Classifier[C, V]) {
	classifier.probOptions = append(classifier.probOptions, prob.OptionLaplaceSmoothing[V])
	classifier.minProb = -1
}

func OptionLogScore[C, V cmp.Ordered](classifier *Classifier[C, V]) {
	classifier.scorer = classifier.logScore
}

func OptionProbScore[C, V cmp.Ordered](classifier *Classifier[C, V]) {
	classifier.scorer = classifier.probScore
}
