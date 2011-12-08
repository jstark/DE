#!/usr/bin/env python
"""
Copyright (c) 2011 John S. Fourkiotis 

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random
import math
import argparse

#---------------------------M O D U L E  C L A S S E S-------------------------
# Problem;
# Objective function definition
class Problem:
	def __init__(self, n, lbounds, ubounds, fun):
		self.n = n
		self.lbounds = lbounds
		self.ubounds = ubounds
		self.fun = fun
	
	def eval(self, x, data = None):
		con = self.constrain(x, data)
		return math.fabs(self.fun(x, data) * con) # NOTE: eval implementation might not suit all problems
	
	def constrain(self, x, data):
		violation = 1
		for i in range(len(x)):
			if x[i] < self.lbounds[i]:
				violation = violation + math.fabs(x[i] - self.lbounds[i])
			if x[i] > self.ubounds[i]:
				violation = violation + math.fabs(x[i] - self.ubounds[i])
		return violation

# Termination;
# Criteria to termination evolution
class Termination:
	def __init__(self, g):
		self.g = g
	
	def maxIterationsReached(self, gen):
		return self.g == gen

# Agent;
# A design vector
class Agent:
	def __init__(self, n):
		self.x = n * [0]
	
	def clone(self):
		a = Agent(len(self.x))
		a.x = [y for y in self.x]
		return a

	def randomize(self, lb, ub):
		self.x = [ lb[i] + (ub[i]-lb[i]) * random.random() for i in range(len(self.x))]

	def __repr__(self):
		return repr(self.x)
	
# Population;
# Agent population of 'NP' agents
class Population:
	def __init__(self, NP, n):
		self.p = [ Agent(n) for i in range(NP) ]

	def random_agent(self):
		r = random.randint(0, len(self.p)-1)
		return self.p[r]
	
	def randomize(self, lb, ub):
		for i in range(len(self.p)):
			self.p[i].randomize(lb, ub)

# DiffEvolutionProgram;
# Differential evolution controller class
class DiffEvolutionParams:
	def __init__(self, population, problem, termination, CR = 0.2, F = 0.8):
		self.population = population
		self.problem = problem
		self.termination = termination
		self.CR = CR
		self.F = F
		self.population.randomize(self.problem.lbounds, self.problem.ubounds)

	def run(self):
		gen = 1
		while self.termination.maxIterationsReached(gen) == False:
			self.evolve()
			gen = gen + 1
		
		results = [ (agent, self.problem.eval(agent.x)) for agent in self.population.p ]
		results = sorted(results, key = lambda r: r[1])
		return results
		
	def evolve(self):
		dinstictAgents = []
		for idx, agent in enumerate(self.population.p):
			# get three (3) random agents != current agent 
			while len(dinstictAgents) != 3:
				random_agent = self.population.random_agent()
				if random_agent != agent and dinstictAgents.count(random_agent) == 0:
					dinstictAgents.append(random_agent)
			
			clonedAgent = agent.clone()
			for i in range(self.problem.n):
				randomIndex = random.randint(0, self.problem.n-1)
				randomUnifr = random.random()
				if randomUnifr < self.CR or i == randomIndex-1:
					a, b, c = dinstictAgents
					clonedAgent.x[i] = a.x[i] + self.F * (b.x[i] - c.x[i])
				else:
					clonedAgent.x[i] = agent.x[i]
			
			if self.problem.eval(clonedAgent.x) < self.problem.eval(agent.x):
				agent.x = clonedAgent.x	

#-------------------------------B E N C H M A R K S----------------------------
# sphere2;
# benchmark problem #1
# f(x,y) = x^2 + y^2
def sphere2(x, data = None):
	return sum([y*y for y in x])

# rosenbrock;
# benchmark problem #2
# f(x,y) = (1-x)^2 + 100(y-x^2)^2
def rosenbrock2(x, data = None):
	return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

# himmelblau;
# benchmark problem #3
# f(x,y) = (x^2 + y - 11)^2 + (y^2 + x - 7)^2
def himmelblau(x, data = None):
    return (x[0]**2+x[1]-11)**2+(x[1]**2+x[0]-7)**2

# schwefel2
# benchmark problem #4
# f(x,y) = 418.9829 * 2 + x sin (sqrt(abs(x))) + y sin (sqrt(abs(x)))
def schwefel2(x, data = None):
    return 418.9829 * 2 + sum([y * math.sin(math.sqrt(math.fabs(y))) for y in x ])

BENCHMARKS = { 
    'sphere2' : Problem(2, [0, 0], [10, 10], sphere2),
    'rosenbrock2' : Problem(2, [0, 0], [10, 10], rosenbrock2), 
    'himmelblau' : Problem(2, [0, 0], [10, 10], himmelblau),
    'schwefel2' : Problem(2, 2 * [-512.03], 2 * [511.97], schwefel2)
}

# benchmark_problem;
# factory method that creates a 'problem' object
# for a given (string) benchmark name
def benchmark_problem(f):
    return BENCHMARKS[f]

#---------------------------D R I V E R  M E T H O D-------------------------
# de;
# driver function	
def de():
	parser = argparse.ArgumentParser(description = 'A Differential Evolution Optimizer program')
	parser.add_argument('--CR', type=float, metavar='number', default=0.2, help='Crossover ratio (default 0.2)')
	parser.add_argument('--g' , type=int  , metavar='iterations' , default=1000, help='Number of generations (default 1000)')
	parser.add_argument('--bench', default='rosenbrock2', help='The benchmark/problem to be run (try "sphere2" or "rosenbrock2"')
	parser.add_argument('--Ps', type=int, metavar='size', default=40, help='Population size (default 40)')

	args = parser.parse_args()
	f = args.bench
	problem = benchmark_problem(f)
	if problem is None:
		print "Benchmark '%s' not found!" % f
		return

	tcriter = Termination(g = args.g)
	population = Population(NP = args.Ps, n = problem.n)
	algorithm = DiffEvolutionParams(population, problem, tcriter, args.CR)
	results = algorithm.run()
	print 'Best solution (X, Fitness):'
	print results[0]

if __name__ == "__main__":
	de()
