package bayes_test

import (
	"fmt"
	"maps"

	"github.com/xuender/aime/bayes"
)

func ExampleClassifier() {
	classifier := bayes.NewClassifier[string, int]()

	classifier.Train(maps.All(map[string][]int{
		"a": {7, 8, 9, 6},
		"c": {1, 2, 3, 4},
		"b": {4, 5, 6, 7},
	}))

	for key, val := range classifier.Scores([]int{1, 2, 3}) {
		fmt.Println(key, val)
	}

	// Output:
	// c -1.0986122886681098
	// a -77.08392035747161
	// b -77.08392035747161
}

func ExampleClassifier_probScore() {
	classifier := bayes.NewClassifier[string, int](bayes.OptionProbScore)

	classifier.Train(maps.All(map[string][]int{
		"a": {7, 8, 9, 6},
		"c": {1, 2, 3, 4},
		"b": {4, 5, 6, 7},
	}))

	for key, val := range classifier.Scores([]int{1, 2, 3}) {
		fmt.Println(key, val)
	}

	// Output:
	// c 1
	// a 9.999999999999999e-34
	// b 9.999999999999999e-34
}

func ExampleClassifier_laplaceSmoothing() {
	classifier := bayes.NewClassifier[string, int](bayes.OptionLaplaceSmoothing)

	classifier.Train(maps.All(map[string][]int{
		"a": {7, 8, 9, 6},
		"c": {1, 2, 3, 4},
		"b": {4, 5, 6, 7},
	}))

	for key, val := range classifier.Scores([]int{1, 2, 3}) {
		fmt.Println(key, val)
	}

	// Output:
	// c -0.5516476182862461
	// a -77.08392035747161
	// b -77.08392035747161
}

func ExampleClassifier_Predict_laplaceSmoothing() {
	classifier := bayes.NewClassifier[string, int](bayes.OptionLaplaceSmoothing)

	classifier.Train(maps.All(map[string][]int{
		"a": {7, 8, 9, 6},
		"c": {1, 2, 3, 4},
		"b": {4, 5, 6, 7},
	}))
	fmt.Println(classifier.Predict([]int{1, 2, 3}))

	// Output:
	// c true
}

func ExampleClassifier_Predict_laplaceSmoothing_false() {
	classifier := bayes.NewClassifier[string, int](bayes.OptionLaplaceSmoothing)

	classifier.Train(maps.All(map[string][]int{
		"a": {7, 8, 9, 6},
		"c": {1, 2, 3, 4},
		"b": {4, 5, 6, 7},
	}))
	fmt.Println(classifier.Predict([]int{1, 2, 3, 4, 5}))

	// Output:
	// c false
}

func ExampleClassifier_Predict() {
	classifier := bayes.NewClassifier[string, int]()

	classifier.Train(maps.All(map[string][]int{
		"a": {7, 8, 9, 6},
		"c": {1, 2, 3, 4},
		"b": {4, 5, 6, 7},
	}))
	fmt.Println(classifier.Predict([]int{1, 2, 3}))

	// Output:
	// c true
}
