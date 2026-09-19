# PLAN
<hr>
## For the Engine:

- Implement TODOs found the in code
- Start building framework for connecting to an external app
- Only think about really expanding the engine once the front end infra is done. That way, the engine actually exists in a good state.
  Otherwise, it could risk feature bloat without a strong core. Don't avoid working on the important stuff to add cool features. 

## Front End:
- Research different ways of constructing a lightweight, clean and fast front end. This could be its own separate app,
  or be simply hosted on a web browser like node-red. Web browser would be easier, however a standalone app would be cool.
  Need to weight maintenance costs and infrastructure bloat before commiting. 
- Figure out how decoupled the engine and the front end should be. Ideally, there should be a large amount of separation
  between them to make changing either easier. 
- On that note, coming up with an expandable, simple, and powerful communication method between the engine and the front-
  end is going to a challenge in of itself. 

  
## Order of Operations:

1. Complete TODOs in the code
2. Decide on the type of frontend that is going to be implemented
3. Refactor engine in a way that doesn't lock it down (haven't done frontend before, will def need to make changed to back end alongside it)
4. Do a full application rigorous test, do some polishing and refactoring to make engine and frontend easily expandable
5. When all that is done, then more (cool) features can be added like rudimentary neural stuff. 