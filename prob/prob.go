package prob

import (
	"cmp"

	"github.com/xuender/aime/pb"
)

const _defaultProb = 0.00000000001

type Prob[V cmp.Ordered] struct {
	count  map[V]float64
	total  float64
	sum    float64
	prober func(V) float64
}

func NewProb[V cmp.Ordered](opts ...Option[V]) *Prob[V] {
	ret := &Prob[V]{
		count: map[V]float64{},
	}

	ret.prober = ret.defaultProb

	for _, opt := range opts {
		opt(ret)
	}

	return ret
}

func Load[V cmp.Ordered](input *pb.Prob) *Prob[V] {
	ret := NewProb[V]()
	ret.total = input.GetTotal()
	ret.sum = input.GetSum()

	// var zero V

	// switch any(zero).(type) {
	// case int32:
	// 	for idx, val := range input.CountInt32 {
	// 		ret.count[V(val)] = input.Values[idx]
	// 	}
	// case int64:
	// 	for idx, val := range input.CountInt64 {
	// 		ret.count[V(val)] = input.Values[idx]
	// 	}
	// case uint32:
	// 	for idx, val := range input.CountUint32 {
	// 		ret.count[V(val)] = input.Values[idx]
	// 	}
	// case uint64:
	// 	for idx, val := range input.CountUint32 {
	// 		ret.count[V(val)] = input.Values[idx]
	// 	}
	// case float32:
	// 	for idx, val := range input.CountFloat {
	// 		ret.count[V(val)] = input.Values[idx]
	// 	}
	// case float64:
	// 	for idx, val := range input.CountDouble {
	// 		ret.count[V(val)] = input.Values[idx]
	// 	}
	// case string:
	// 	for idx, val := range input.CountString {
	// 		ret.count[V(val)] = input.Values[idx]
	// 	}
	// }

	return ret
}

func (p *Prob[V]) Total() float64 {
	return p.total
}

func (p *Prob[V]) Add(items []V) {
	for _, item := range items {
		p.count[item]++
	}

	p.sum += float64(len(items))
	p.total++
}

func (p *Prob[V]) Prob(val V) float64 {
	return p.prober(val)
}

func (p *Prob[V]) defaultProb(val V) float64 {
	count, has := p.count[val]
	if !has {
		return _defaultProb
	}

	return count / float64(p.total)
}

func (p *Prob[V]) laplaceSmoothing(val V) float64 {
	count, has := p.count[val]
	if has {
		return count + 1/(p.sum+p.total)
	}

	return _defaultProb
}

func (p *Prob[V]) Proto() *pb.Prob {
	msg := &pb.Prob{
		Total: p.total,
		Sum:   p.sum,
	}

	if len(p.count) == 0 {
		return msg
	}

	for key, count := range p.count {
		msg.Values = append(msg.Values, count)

		switch val := any(key).(type) {
		case int32:
			msg.CountInt32 = append(msg.CountInt32, val)
		case int64:
			msg.CountInt64 = append(msg.CountInt64, val)
		case uint32:
			msg.CountUint32 = append(msg.CountUint32, val)
		case uint64:
			msg.CountUint64 = append(msg.CountUint64, val)
		case float32:
			msg.CountFloat = append(msg.CountFloat, val)
		case float64:
			msg.CountDouble = append(msg.CountDouble, val)
		case string:
			msg.CountString = append(msg.CountString, val)
		}
	}

	return msg
}
