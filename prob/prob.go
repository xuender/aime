package prob

import "cmp"

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
