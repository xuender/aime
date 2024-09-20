package bayes

import (
	"cmp"
	"iter"
	"maps"
	"math"

	"github.com/xuender/aime/prob"
	"github.com/xuender/flow"
	"github.com/xuender/flow/seq"
)

// nolint: gochecknoglobals
var _none = struct{}{}

type Classifier[C, V cmp.Ordered] struct {
	probabilities map[C]float64
	frequencies   map[C]*prob.Prob[V]
	vocabulary    map[V]struct{}
	learned       float64
	probOptions   []prob.Option[V]
	minProb       float64
}

func NewClassifier[C, V cmp.Ordered](opts ...Option[C, V]) *Classifier[C, V] {
	ret := &Classifier[C, V]{
		probabilities: map[C]float64{},
		frequencies:   map[C]*prob.Prob[V]{},
		vocabulary:    map[V]struct{}{},
		probOptions:   []prob.Option[V]{},
		minProb:       -2,
	}

	for _, opt := range opts {
		opt(ret)
	}

	return ret
}

func (p *Classifier[C, V]) Train(seq iter.Seq2[C, []V]) {
	for class, items := range seq {
		p.learned++

		freq := p.frequencies[class]
		if freq == nil {
			freq = prob.NewProb[V](p.probOptions...)
			p.frequencies[class] = freq
		}

		freq.Total++

		for _, item := range items {
			p.vocabulary[item] = _none

			freq.Add(item)
		}
	}

	for class, freq := range p.frequencies {
		p.probabilities[class] = freq.Total / p.learned
	}
}

func (p *Classifier[C, V]) Scores(items []V) iter.Seq2[C, float64] {
	num := len(p.probabilities)
	scores := make(map[C]float64, num)

	for class, prob := range p.probabilities {
		score := math.Log(prob)
		fre := p.frequencies[class]

		for _, item := range items {
			score += math.Log(fre.Prob(item))
		}

		scores[class] = score
	}

	return flow.Chain2(
		maps.All(scores),
		flow.SortFunc2(func(t1, t2 seq.Tuple[C, float64]) int {
			ret := int(t2.V - t1.V)
			if ret == 0 {
				return cmp.Compare(t1.K, t2.K)
			}

			return ret
		}),
	)
}

func (p *Classifier[C, V]) Predict(items []V) (C, bool) {
	class, prod, has := seq.Reduce2(flow.Chain2(
		maps.All(p.probabilities),
		flow.Map2(func(class C, val float64) (C, float64) {
			score := math.Log(val)

			for _, item := range items {
				smoothedFreq := p.frequencies[class].Prob(item)
				score += math.Log(smoothedFreq)
			}

			return class, score
		}),
	), func(class1 C, score1 float64, class2 C, score2 float64) (C, float64) {
		if score1 > score2 {
			return class1, score1
		}

		return class2, score2
	})

	if prod < p.minProb {
		return class, false
	}

	return class, has
}
