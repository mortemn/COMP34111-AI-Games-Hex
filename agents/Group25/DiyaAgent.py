# agents/Group25/MCTSAgent.py
from __future__ import annotations

import time
import math
import random
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

from src.AgentBase import AgentBase
from src.Move import Move
from src.Colour import Colour


# ---------------------------------------------------------
# FAST BOARD (Union-Find for win detection)
# ---------------------------------------------------------
class BoardState:
    EMPTY = 0
    RED = 1
    BLUE = 2
    NEIGH = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]

    def __init__(self, size=11):
        self.size = size
        self.grid = [[BoardState.EMPTY]*size for _ in range(size)]
        self.empty = [(x,y) for x in range(size) for y in range(size)]
        n = size*size
        self.red = list(range(n+2))
        self.blue = list(range(n+2))
        self.TOP, self.BOTTOM = n, n+1
        self.LEFT, self.RIGHT = n, n+1
        self._winner = None

    def idx(self,x,y): return x*self.size+y

    def find(self, p, a):
        while p[a]!=a:
            p[a]=p[p[a]]
            a=p[a]
        return a

    def union(self,p,a,b):
        ra,rb=self.find(p,a),self.find(p,b)
        if ra!=rb: p[rb]=ra

    def copy(self):
        new = BoardState(self.size)
        new.grid = [row[:] for row in self.grid]
        new.empty = list(self.empty)
        new.red = self.red[:]
        new.blue = self.blue[:]
        new._winner = self._winner
        return new

    def play(self,x,y,c):
        if self.grid[x][y]!=BoardState.EMPTY:
            raise ValueError("Illegal move")
        self.grid[x][y]=c
        if (x,y) in self.empty: self.empty.remove((x,y))

        idx=self.idx(x,y)
        if c==BoardState.RED:
            p=self.red
            if x==0: self.union(p,idx,self.TOP)
            if x==self.size-1: self.union(p,idx,self.BOTTOM)
            for dx,dy in BoardState.NEIGH:
                nx,ny=x+dx,y+dy
                if 0<=nx<self.size and 0<=ny<self.size and \
                   self.grid[nx][ny]==BoardState.RED:
                    self.union(p,idx,self.idx(nx,ny))
            if self.find(p,self.TOP)==self.find(p,self.BOTTOM):
                self._winner=Colour.RED

        else:
            p=self.blue
            if y==0: self.union(p,idx,self.LEFT)
            if y==self.size-1: self.union(p,idx,self.RIGHT)
            for dx,dy in BoardState.NEIGH:
                nx,ny=x+dx,y+dy
                if 0<=nx<self.size and 0<=ny<self.size and \
                   self.grid[nx][ny]==BoardState.BLUE:
                    self.union(p,idx,self.idx(nx,ny))
            if self.find(p,self.LEFT)==self.find(p,self.RIGHT):
                self._winner=Colour.BLUE

    def winner(self):
        return self._winner

    @staticmethod
    def from_engine(board):
        bs = BoardState(board.size)
        bs.empty=[]
        n=board.size*board.size
        bs.red,bs.blue=list(range(n+2)),list(range(n+2))

        for x in range(bs.size):
            for y in range(bs.size):
                t=board.tiles[x][y].colour
                if t is None:
                    bs.grid[x][y]=BoardState.EMPTY
                    bs.empty.append((x,y))
                else:
                    bs.grid[x][y] = BoardState.RED if t==Colour.RED else BoardState.BLUE

        for x,y in [(i,j) for i in range(bs.size) for j in range(bs.size)]:
            c=bs.grid[x][y]
            if c==BoardState.EMPTY: continue
            idx=bs.idx(x,y)
            if c==BoardState.RED:
                if x==0: bs.union(bs.red,idx,bs.TOP)
                if x==bs.size-1: bs.union(bs.red,idx,bs.BOTTOM)
            else:
                if y==0: bs.union(bs.blue,idx,bs.LEFT)
                if y==bs.size-1: bs.union(bs.blue,idx,bs.RIGHT)

        return bs


# ---------------------------------------------------------
# SIMPLE HEURISTIC FOR ROLLOUTS
# ---------------------------------------------------------
def rollout_move(st:BoardState, c:int):
    opp = BoardState.RED if c==BoardState.BLUE else BoardState.BLUE
    own=[]
    block=[]
    for x,y in st.empty:
        near=0;nearOpp=0
        for dx,dy in BoardState.NEIGH:
            nx,ny=x+dx,y+dy
            if 0<=nx<st.size and 0<=ny<st.size:
                if st.grid[nx][ny]==c: near+=1
                if st.grid[nx][ny]==opp: nearOpp+=1
        if near>0: own.append((x,y))
        elif nearOpp>0: block.append((x,y))
    if own: return random.choice(own)
    if block: return random.choice(block)
    return random.choice(st.empty)


# ---------------------------------------------------------
# MCTS NODE
# ---------------------------------------------------------
class Node:
    def __init__(self, st:BoardState, p:Colour, parent=None, mv=None, root=None):
        self.st=st
        self.p=p
        self.parent=parent
        self.mv=mv
        self.root=root if root else p
        self.ch=[]
        self.untried=list(st.empty)
        self.w=0; self.v=0
        self.C=math.sqrt(2)

    def ucb(self,c):
        if c.v==0: return float("inf")
        return (c.w/c.v) + self.C*math.sqrt(math.log(self.v+1)/(c.v))

    def select(self):
        n=self
        while not n.untried and n.ch:
            n=max(n.ch,key=lambda x:self.ucb(x))
        return n

    def expand(self):
        x,y = self.untried.pop()
        new = self.st.copy()
        new.play(x,y, BoardState.RED if self.p==Colour.RED else BoardState.BLUE)
        nxt = Node(new,Colour.opposite(self.p),self,Move(x,y),self.root)
        self.ch.append(nxt)
        return nxt

    def rollout(self):
        st=self.st.copy()
        p=self.p
        while True:
            if st.winner(): return st.winner()
            if not st.empty: return Colour.opposite(p)
            c = BoardState.RED if p==Colour.RED else BoardState.BLUE
            x,y = rollout_move(st,c)
            st.play(x,y,c)
            p=Colour.opposite(p)

    def back(self,res):
        self.v+=1
        if res==self.root: self.w+=1
        if self.parent: self.parent.back(res)

    def run(self,iters,maxt=None):
        t=time.time()
        for _ in range(iters):
            if maxt and time.time()-t>maxt: break
            n=self.select()
            if n.untried: n=n.expand()
            res=n.rollout()
            n.back(res)


# ---------------------------------------------------------
# AGENT
# ---------------------------------------------------------
class MCTSAgent(AgentBase):
    def __init__(self,c):
        super().__init__(c)
        self.iters=20000
        self.time=2.0
        self.workers=2
        self.debug=True

    def worker(self,st):
        rt=Node(st.copy(),self.colour)
        rt.run(self.iters//self.workers,self.time/self.workers)
        stats={}
        for c in rt.ch:
            stats[(c.mv.x,c.mv.y)] = (c.v,c.w)
        return stats

    def make_move(self,t,b,om):
        st=BoardState.from_engine(b)
        if not st.empty: return Move(0,0)

        start=time.time()
        agg={}

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            fs=[ex.submit(self.worker,st) for _ in range(self.workers)]
            for f in fs:
                for m,(v,w) in f.result().items():
                    agg[m]=[agg.get(m,[0,0])[0]+v,
                            agg.get(m,[0,0])[1]+w]

        if not agg: mx,my=random.choice(st.empty)
        else: mx,my=max(agg.items(),key=lambda x:x[1][0])[0]

        if self.debug:
            print(f"[MCTS] search {time.time()-start:.2f}s")

        return Move(mx,my)
