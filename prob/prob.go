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

// nolint
func Load[V cmp.Ordered](input *pb.Prob) *Prob[V] {
	ret := NewProb[V]()
	ret.total = input.GetTotal()
	ret.sum = input.GetSum()

	var zero V

	switch any(zero).(type) {
	case int32:
		for idx, val := range input.GetValues() {
			ret.count[any(input.GetCountInt32()[idx]).(V)] = val
		}
	case int64:
		for idx, val := range input.GetValues() {
			ret.count[any(input.GetCountInt64()[idx]).(V)] = val
		}
	case uint32:
		for idx, val := range input.GetValues() {
			ret.count[any(input.GetCountUint32()[idx]).(V)] = val
		}
	case uint64:
		for idx, val := range input.GetValues() {
			ret.count[any(input.GetCountUint64()[idx]).(V)] = val
		}
	case float32:
		for idx, val := range input.GetValues() {
			ret.count[any(input.GetCountFloat()[idx]).(V)] = val
		}
	case float64:
		for idx, val := range input.GetValues() {
			ret.count[any(input.GetCountDouble()[idx]).(V)] = val
		}
	case string:
		for idx, val := range input.GetValues() {
			ret.count[any(input.GetCountString()[idx]).(V)] = val
		}
	}

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

// nolint
func (p *Prob[V]) Proto() *pb.Prob {
	msg := &pb.Prob{
		Total: p.total,
		Sum:   p.sum,
	}

	if len(p.count) == 0 {
		return msg
	}

	var zero V

	switch any(zero).(type) {
	case int32:
		for key, count := range p.count {
			msg.Values = append(msg.Values, count)
			msg.CountInt32 = append(msg.CountInt32, any(key).(int32))
		}
	case int64:
		for key, count := range p.count {
			msg.Values = append(msg.Values, count)
			msg.CountInt64 = append(msg.CountInt64, any(key).(int64))
		}
	case uint32:
		for key, count := range p.count {
			msg.Values = append(msg.Values, count)
			msg.CountUint32 = append(msg.CountUint32, any(key).(uint32))
		}
	case uint64:
		for key, count := range p.count {
			msg.Values = append(msg.Values, count)
			msg.CountUint64 = append(msg.CountUint64, any(key).(uint64))
		}
	case float32:
		for key, count := range p.count {
			msg.Values = append(msg.Values, count)
			msg.CountFloat = append(msg.CountFloat, any(key).(float32))
		}
	case float64:
		for key, count := range p.count {
			msg.Values = append(msg.Values, count)
			msg.CountDouble = append(msg.CountDouble, any(key).(float64))
		}
	case string:
		for key, count := range p.count {
			msg.Values = append(msg.Values, count)
			msg.CountString = append(msg.CountString, any(key).(string))
		}
	}

	return msg
}
