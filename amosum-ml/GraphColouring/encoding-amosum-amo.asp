{col(X,C) : colour(C)} <= 1 :- node(X).
:- col(X1,C), col(X2,C), link(X1,X2).
#amosum{ W: col(X,C), colour_weight(C,W) [X]} >= MIN : lb(MIN,0).