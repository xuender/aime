package aime

import "iter"

type Classifier[C, V comparable] interface {
	Train(seq iter.Seq2[C, []V])
	Scores(values []V) iter.Seq2[C, float64]
	Predict(values []V) (C, bool)
}
