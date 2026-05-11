% ENCODING
% --------------------------------------------------------------------------------------------
salary(full_professor,70).
salary(associate_professor,50).
salary(researcher,30).
max_hours_working_per_month(125).
month(1..12).
possible_hours(40).
possible_hours(60).
possible_hours(80).
possible_hours(100).
possible_hours(125).
:- person(Per), month(M), max_hours_working_per_month(MAX), #sum{H,Pro: join_project(Per, Pro, H, M)} > MAX.
:- project(Pro, _, UB), #sum{ H*S, Per, M: join_project(Per, Pro, H, M), role(Per,R), salary(R,S), project_month(Pro, SM, EM), M >= SM, M <= EM} > UB.

% amo
{join_project(Per,Pro, H, M) : possible_hours(H)} <= 1 :- person(Per), project(Pro, LB, UB), month(M), project_month(Pro, SM, EM), M >= SM, M <= EM.

% sum
:- project(Pro, LB, UB), #sum{ H*S, Per, M: join_project(Per, Pro, H, M), role(Per,R), salary(R,S), project_month(Pro, SM, EM), M >= SM, M <= EM} < LB.





