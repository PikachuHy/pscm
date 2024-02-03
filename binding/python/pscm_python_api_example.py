import pypscm

scm = pypscm.Scheme()
print(scm)
ret = scm.eval("(+ 2 6)")
print(ret)
