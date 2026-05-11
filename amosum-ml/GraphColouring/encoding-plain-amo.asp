{col(X,C) : colour(C)} <= 1 :- node(X).
:- col(X1,C), col(X2,C), link(X1,X2).
:- #sum{W,X: col(X,C), colour_weight(C,W)} < B, lb(B,0).