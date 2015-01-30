CC = gcc
CXX = g++
FLAGS = -O3 -g

all: loopsize

warpnum: parser_warpnum_atstalls.cpp
	$(CXX) $(FLAGS) parser_warpnum_atstalls.cpp -o warpnum_atstall

loopsize: parser_loopsize_perthread.cpp
	$(CXX) $(FLAGS) parser_loopsize_perthread.cpp -o loopsize_perthread 

clean:
	rm -rf warpnum_atstall loopsize_perthread
