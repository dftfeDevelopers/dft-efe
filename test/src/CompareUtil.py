import numpy as np

class Compare():
    def cmp(self, val1, val2, tol = 1.0e-16, cmpType = 'absolute', normType="L2"):
        msg = ""
        areComparable = False
        v1 = np.array(val1)
        v2 = np.array(val2)
        norm = 0.0
        if len(val1) != len(val2):
            msg="The two values to be compared are of different sizes"
        
        elif normType=='L1':
            diff = v1-v2
            norm = np.linalg.norm(diff, 1)
            if cmpType=='relative':
                norm = norm/np.linalg.norm(v2, 1)
            
            areComparable = norm < tol
            
        elif normType=='L2':
            diff = v1-v2
            norm = np.linalg.norm(diff)
            if cmpType=='relative':
                norm = norm/np.linalg.norm(v2)
            
            areComparable = norm < tol
            
        elif normType=='inf':
            diff = v1-v2
            norm = np.linalg.norm(diff, 'inf')
            if cmpType=='relative':
                norm = norm/np.linalg.norm(v2, 'inf')
            
            areComparable = norm < tol
            
        else:
            msg = "Invalid normType passed"
            areComparable = False

        if not areComparable and msg!="":
            msg = "The two values passed are not comparable. Norm of \
                    difference={diff_norm} and tolerance={tol}".format(diff_norm=norm,
                                                                       tol=tol)
        return areComparable, msg

    def cmpElementWise(val1, val2, tol = 1.0e-16, cmpType = 'actual'):
        msg = ""
        v1 = np.array(val1)
        v2 = np.array(val2)
        norm = 0.0
        maxVal = 0.0
        if len(val1) != len(val2):
            msg="The two values to be compared are of different sizes"
            areComparable = False
        else:
            diff = np.abs(v1 - v2)
            if cmpType=='relative':
                diff = diff/np.abs(v2)

            areComparable = (diff < tol).all()
            maxVal = np.amax(diff)

        if not areComparable and msg!="":
            msg = "The two values passed are not comparable. Max. of \
                    elementwise difference={maxVal} and \
                    tolerance={tol}".format(maxVal=maxVal,tol=tol)

        return areComparable, msg

