Places to look for issues:

- Model objective: does this include all necessary parts?  
    x first-stage x costs
    x first-stage u costs
    x second-stage expectation costs
    x second-stage CVaR costs
- CVaR lower bound ok?
- CVaR PosPartRule correctly defined? 
    - should be CvarPosPart >= ScenObjValue - CvarAux
- ScenObjValue correctly defined? 
    - should be q*y
    > Is equal to c*x + q*y
- obj_fun_mean_CVaR correctly defined?
    - should be c*x + lambda*u + (1-lambda)*ScenObjValue + lambda/(1-beta) * CvarPosPart

- in scenario settings:
    - first-stage objective correctly defined?
        - should be c*x + lambda*u
    - first-stage variables correctly defined?
        - should be [x] + [CvarAux]


##############

Probably split objective in first-stage and second-stage. That should help.