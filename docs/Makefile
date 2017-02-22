all:

../README.md: src/tutorial.md
	julia makemd.jl
	perl mkreadme.pl build/tutorial.md > ../README.md

exampledocs: getdoc.pl
	echo "# Example docstrings" > exampledocs.jl
	for f in `ls ../examples/*.jl`; do perl getdoc.pl $$f >> exampledocs.jl; done
