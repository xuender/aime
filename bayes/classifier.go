package bayes

import (
	"cmp"
	"iter"
	"maps"
	"math"

	"github.com/xuender/aime/pb"
	"github.com/xuender/aime/prob"
	"github.com/xuender/flow"
	"github.com/xuender/flow/seq"
)

// nolint: gochecknoglobals
var _none = struct{}{}

type Classifier[C, V cmp.Ordered] struct {
	// Prior Probability.
	prior       map[C]float64
	probs       map[C]*prob.Prob[V]
	values      map[V]struct{}
	learned     float64
	probOptions []prob.Option[V]
	minProb     float64
	scorer      func([]V, map[C]float64)
}

func NewClassifier[C, V cmp.Ordered](opts ...Option[C, V]) *Classifier[C, V] {
	ret := &Classifier[C, V]{
		prior:       map[C]float64{},
		probs:       map[C]*prob.Prob[V]{},
		values:      map[V]struct{}{},
		probOptions: []prob.Option[V]{},
		minProb:     -2,
	}

	ret.scorer = ret.logScore

	for _, opt := range opts {
		opt(ret)
	}

	return ret
}

func (p *Classifier[C, V]) Learned() float64 {
	return p.learned
}

func Load[C, V cmp.Ordered](input *pb.Classifier) *Classifier[C, V] {
	var (
		zeroV V
		zeroC C
	)

	ret := NewClassifier[C, V]()
	ret.learned = input.GetLearned()
	ret.minProb = input.MinProb

	switch any(zeroV).(type) {
	case int32:
		for _, val := range input.GetValueInt32() {
			ret.values[any(val).(V)] = _none
		}
	case int64:
		for _, val := range input.GetValueInt64() {
			ret.values[any(val).(V)] = _none
		}
	case uint64:
		for _, val := range input.GetValueUint64() {
			ret.values[any(val).(V)] = _none
		}
	case float32:
		for _, val := range input.GetValueFloat() {
			ret.values[any(val).(V)] = _none
		}
	case float64:
		for _, val := range input.GetValueDouble() {
			ret.values[any(val).(V)] = _none
		}
	case string:
		for _, val := range input.GetValueString() {
			ret.values[any(val).(V)] = _none
		}
	}

	switch any(zeroC).(type) {
	case int32:
		for idx, class := range input.GetClassInt32() {
			key := any(class).(C)
			ret.prior[key] = input.GetPrior()[idx]
			ret.probs[key] = prob.Load[V](input.GetProb()[idx])
		}
	case int64:
		for idx, class := range input.GetClassInt64() {
			key := any(class).(C)
			ret.prior[key] = input.GetPrior()[idx]
			ret.probs[key] = prob.Load[V](input.GetProb()[idx])
		}
	case uint64:
		for idx, class := range input.GetClassUint64() {
			key := any(class).(C)
			ret.prior[key] = input.GetPrior()[idx]
			ret.probs[key] = prob.Load[V](input.GetProb()[idx])
		}
	case float32:
		for idx, class := range input.GetClassFloat() {
			key := any(class).(C)
			ret.prior[key] = input.GetPrior()[idx]
			ret.probs[key] = prob.Load[V](input.GetProb()[idx])
		}
	case float64:
		for idx, class := range input.GetClassDouble() {
			key := any(class).(C)
			ret.prior[key] = input.GetPrior()[idx]
			ret.probs[key] = prob.Load[V](input.GetProb()[idx])
		}
	case string:
		for idx, class := range input.GetClassString() {
			key := any(class).(C)
			ret.prior[key] = input.GetPrior()[idx]
			ret.probs[key] = prob.Load[V](input.GetProb()[idx])
		}
	}
	return ret
}

func (p *Classifier[C, V]) Train(seq iter.Seq2[C, []V]) {
	for class, items := range seq {
		p.learned++

		val := p.probs[class]
		if val == nil {
			val = prob.NewProb[V](p.probOptions...)
			p.probs[class] = val
		}

		val.Add(items)

		for _, item := range items {
			p.values[item] = _none
		}
	}

	for class, val := range p.probs {
		p.prior[class] = val.Total() / p.learned
	}
}

func (p *Classifier[C, V]) Scores(items []V) iter.Seq2[C, float64] {
	num := len(p.prior)
	scores := make(map[C]float64, num)

	p.scorer(items, scores)

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

func (p *Classifier[C, V]) logScore(items []V, scores map[C]float64) {
	for class, score := range p.prior {
		val := p.probs[class]
		score = math.Log(score)

		for _, item := range items {
			score += math.Log(val.Prob(item))
		}

		scores[class] = score
	}
}

func (p *Classifier[C, V]) probScore(items []V, scores map[C]float64) {
	var sum float64

	for class, score := range p.prior {
		val := p.probs[class]

		for _, item := range items {
			score *= val.Prob(item)
		}

		scores[class] = score
		sum += score
	}

	for class := range scores {
		scores[class] /= sum
	}
}

func (p *Classifier[C, V]) Predict(items []V) (C, bool) {
	class, prod, has := seq.Reduce2(flow.Chain2(
		maps.All(p.prior),
		flow.Map2(func(class C, val float64) (C, float64) {
			score := math.Log(val)

			for _, item := range items {
				smoothedFreq := p.probs[class].Prob(item)
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

// nolint: cyclop
func (p *Classifier[C, V]) Proto() *pb.Classifier {
	msg := &pb.Classifier{
		Learned: p.learned,
		MinProb: p.minProb,
	}

	for class, prior := range p.prior {
		msg.Prior = append(msg.Prior, prior)
		msg.Prob = append(msg.Prob, p.probs[class].Proto())

		switch val := any(class).(type) {
		case int32:
			msg.ClassInt32 = append(msg.ClassInt32, val)
		case int64:
			msg.ClassInt64 = append(msg.ClassInt64, val)
		case uint32:
			msg.ClassUint32 = append(msg.ClassUint32, val)
		case uint64:
			msg.ClassUint64 = append(msg.ClassUint64, val)
		case float32:
			msg.ClassFloat = append(msg.ClassFloat, val)
		case float64:
			msg.ClassDouble = append(msg.ClassDouble, val)
		case string:
			msg.ClassString = append(msg.ClassString, val)
		}
	}

	for value := range p.values {
		switch val := any(value).(type) {
		case int32:
			msg.ValueInt32 = append(msg.ValueInt32, val)
		case int64:
			msg.ValueInt64 = append(msg.ValueInt64, val)
		case uint32:
			msg.ValueUint32 = append(msg.ValueUint32, val)
		case uint64:
			msg.ValueUint64 = append(msg.ValueUint64, val)
		case float32:
			msg.ValueFloat = append(msg.ValueFloat, val)
		case float64:
			msg.ValueDouble = append(msg.ValueDouble, val)
		case string:
			msg.ValueString = append(msg.ValueString, val)
		}
	}

	return msg
}

// func (p *Classifier[C, V]) Unmarshal(msg *pb.Classifier) {
// 	proto.Marshal()
// 	proto.Unmarshal()
// }
