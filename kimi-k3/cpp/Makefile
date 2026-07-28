CXX ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Iinclude

.PHONY: test clean

test_goldens: src/ops.cpp src/test_goldens.cpp include/kimi_k3_ref.hpp ../fixtures/goldens_embedded.h
	$(CXX) $(CXXFLAGS) -o test_goldens src/ops.cpp src/test_goldens.cpp

test: test_goldens
	./test_goldens

clean:
	rm -f test_goldens
