package prob

import "cmp"

type Option[V cmp.Ordered] func(*Prob[V])

func OptionLaplaceSmoothing[V cmp.Ordered](freq *Prob[V]) {
	freq.prober = freq.laplaceSmoothing
}
