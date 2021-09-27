import numpy as np

class Compare():
    def cmp(self, val1, val2, tol = 1.0e-16, cmpType = 'absolute', normType="L2"):
        msg = ""
        areComparable = False
        v1 = np.array(val1)
        v2 = np.array(val2)
        norm = 0.0
        cmpTypeLower = cmpType.lower()
        isValidCmpType = cmpTypeLower in ["absolute", "relative"]
        normTypeLower = normType.lower()
        isValidNormType = normTypeLower in ["l1", "l2", "inf", "point"]
        if len(val1) != len(val2):
            msg='''The two values to be compared are of different sizes. The
            sizes of first and second list are: {} {}'''. format(len(val1),
                                                                 len(val2))
            raise ValueError(msg)

        if not isValidCmpType:
            msg='''Invalid value in cmpType. The valid values are: 'absolute'
            and 'relative'. The value provided is {}'''. format(cmpType)
            raise ValueError(msg)

        if not isValidNormType:
            msg='''Invalid value in normType. The valid values are: "L2",
            "L1", "inf", and "point". The value provided is {}'''. format(normType)
            raise ValueError(msg)

        if normTypeLower=='l1':
            diff = v1-v2
            norm = np.linalg.norm(diff, 1)
            if cmpTypeLower=='relative':
                norm = norm/np.linalg.norm(v2, 1)
            
            areComparable = norm < tol
            
        if normTypeLower=='l2':
            diff = v1-v2
            norm = np.linalg.norm(diff)
            if cmpTypeLower=='relative':
                norm = norm/np.linalg.norm(v2)
            
            areComparable = norm < tol
            
        if normTypeLower=='inf':
            diff = v1-v2
            norm = np.linalg.norm(diff, 'inf')
            if cmpTypeLower=='relative':
                norm = norm/np.linalg.norm(v2, 'inf')
            
            areComparable = norm < tol
        
        if normTypeLower=='point':
            diff = np.abs(v1 - v2)
            if cmpTypeLower=='relative':
                diff = diff/np.abs(v2)

            norm = np.amax(diff)
            areComparable = (diff < tol).all()

        if not areComparable:
            if normTypeLower in ["l1", "l2", "inf"]:
                msg = "The two values passed are not comparable. Norm of \
                    difference={diff_norm} and tolerance={tol}".format(diff_norm=norm,
                                                                       tol=tol)
            else:
                msg = "The two values passed are not comparable. Max. of \
                    elementwise difference={norm} and \
                    tolerance={tol}".format(norm=norm,tol=tol)

        return areComparable, msg
