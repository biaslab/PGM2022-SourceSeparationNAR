export IdentityMatrix

import Base: * 

struct IdentityMatrix end 

Base.:*(x::IdentityMatrix, y) = return y
Base.:*(y, x::IdentityMatrix) = return y
