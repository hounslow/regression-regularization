import sys
import argparse
import linear_model
import matplotlib.pyplot as plt
import numpy as np
import utils
import os

if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True)
    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        # Load the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2) / n
        print("Training error = ", trainError)

        # Compute test error

        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2) / t
        print ("Test error = ", testError)

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.', label = "Training data")
        plt.title('Training Data')
        # Choose points to evaluate the function
        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g', label = "Least squares fit")
        plt.legend(loc="best")
        figname = os.path.join("..","figs","leastSquares.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "1.1":
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        # Fit least-squares model
        model = linear_model.LeastSquaresBias()
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2) / n
        print("Training error = ", trainError)

        # Compute test error

        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2) / t
        print ("Test error = ", testError)

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.', label = "Training data")
        plt.title('Training Data')

        # Choose points to evaluate the function
        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g', label = "Least squares fit")
        plt.legend(loc="best")
        figname = os.path.join("..","figs","leastSquareswithbias.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "1.2":

        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]

        for p in range(11):
            print("p=%d" % p)

            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)

            # Compute training error
            yhat = model.predict(X)
            trainError = np.sum((yhat - y)**2) / n
            print("Training error = ", trainError)

            # Compute test error

            yhat = model.predict(Xtest)
            testError = np.sum((yhat - ytest)**2) / t
            print ("Test error = ", testError)

            # Plot model
            plt.figure()
            plt.plot(X,y,'b.', label = "Training data")
            plt.title('Training Data. p = {}'.format(p))

            # Choose points to evaluate the function
            Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
            yhat = model.predict(Xhat)

            plt.plot(Xhat,yhat,'g', label = "Least squares fit")
            plt.legend(loc="best")
            figname = os.path.join("..","figs","PolyBasis%d.pdf"%p)
            # print("Saving", figname)
            plt.savefig(figname)


    elif question == "2.1":
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        (n,d) = X.shape
        t = Xtest.shape[0]

        # Split training data into a training and a validation set
        Xtrain = X[0:n//2]
        ytrain = y[0:n//2]
        Xvalid = X[n//2: n]
        yvalid = y[n//2: n]

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set

        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s

            # Train on the training set
            model = linear_model.LeastSquaresRBF(sigma)
            model.fit(Xtrain,ytrain)

            # Compute the error on the validation set
            yhat = model.predict(Xvalid)
            validError = np.sum((yhat - yvalid)**2)/ (n//2)
            print("Error with sigma = {:e} = {}".format( sigma ,validError))

            # Keep track of the lowest validation error
            if validError < minErr:
                minErr = validError
                bestSigma = sigma

            # ** FIX TO 2.1 **
            # # Train on the validation set
            # model = linear_model.LeastSquaresRBF(sigma)
            # model.fit(Xvalid,yvalid)
            #
            # # Compute the error on the training set
            # yhat = model.predict(Xtrain)
            # validError = np.sum((yhat - ytrain)**2)/ (n//2)
            # print("Error with sigma = {:e} = {}".format( sigma ,validError))
            #
            # # Keep track of the lowest validation error
            # if validError < minErr:
            #     minErr = validError
            #     bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error = {:e}".format(bestSigma))

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2)/n
        print("Training error = {}".format(trainError))

        # Finally, report the error on the test set
        t = Xtest.shape[0]
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2)/t
        print("Test error = {}".format(testError))

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title('Training Data')

        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g',label = "Least Squares with RBF kernel and $\sigma={}$".format(bestSigma))
        plt.ylim([-300,400])
        plt.legend()
        figname = os.path.join("..","figs","least_squares_rbf.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "2.2":
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # get the number of rows(n) and columns(d)
        (n,d) = X.shape
        t = Xtest.shape[0]

        m = 9 * n / 10

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set
        minErr = np.inf
        for s in range(-15,16):
            sigma = 2 ** s

            model = linear_model.LeastSquaresRBF(sigma)
            for i in range(0, 9):
                # Train on the training set
                if i == 0:
                    model.fit(X[(i+1)*n//10:],y[(i+1)*n//10:])
                if i >= 1:
                    model.fit(np.vstack((X[0:(i*n)//10], X[(i+1)*n//10:])),np.vstack((y[0:(i*n)//10], y[(i+1)*n//10:])))

                Xvalid = X[(i*n)//10:(i+1)*n//10]
                yvalid = y[(i*n)//10:(i+1)*n//10]
                # Testing on the remaining
                yhat = model.predict(Xvalid)
                validError = np.sum((yhat - yvalid)**2)/ (n//10)

                print("Error with sigma = {:e} = {}".format( sigma ,validError))
                # index = index + 1
                # Keep track of the lowest validation error
                if validError < minErr:
                    minErr = validError
                    bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error = {:e}".format(bestSigma))

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2)/n
        print("Training error = {}".format(trainError))

        # Finally, report the error on the test set
        t = Xtest.shape[0]
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2)/t
        print("Test error = {}".format(testError))

    elif question == "3":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        model = linear_model.logReg(maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogReg Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logReg Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())


    elif question == "3.1":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL2 model
        model = linear_model.logRegL2(lammy=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
        print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "3.2":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL1 model
        model = linear_model.logRegL1(L1_lambda=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nlogRegL1 Training error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("logRegL1 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())


    elif question == "3.3":
        data = utils.load_dataset("logisticData")
        XBin, yBin = data['X'], data['y']
        XBinValid, yBinValid = data['Xvalid'], data['yvalid']

        # Fit logRegL0 model
        model = linear_model.logRegL0(L0_lambda=1.0, maxEvals=400)
        model.fit(XBin,yBin)

        print("\nTraining error %.3f" % utils.classification_error(model.predict(XBin),yBin))
        print("Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
        print("# nonZeros: %d" % (model.w != 0).sum())


    elif question == "4":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # Fit One-vs-all Least Squares
        model = linear_model.leastSquaresClassifier()
        model.fit(XMulti, yMulti)

        print("leastSquaresClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("leastSquaresClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

        print(np.unique(model.predict(XMultiValid)))

    elif question == "4.1":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # Fit One-vs-all Logistic Regression
        model = linear_model.logLinearClassifier(maxEvals=500, verbose=0)
        model.fit(XMulti, yMulti)

        print("logLinearClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("logLinearClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))


    elif question == "4.4":
        data = utils.load_dataset("multiData")
        XMulti, yMulti = data['X'], data['y']
        XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

        # Fit logRegL2 model
        model = linear_model.softmaxClassifier(maxEvals=500)
        model.fit(XMulti, yMulti)

        print("Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
        print("Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

    else:
        print("Not a valid question number.")
