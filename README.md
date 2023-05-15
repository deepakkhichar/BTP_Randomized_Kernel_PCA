The aim of our project is to find the most computationally efficient method that can be used
to store and comprehend a given set of n data points, each lying in a d-dimensional space such
that the number of elements required in its representation and thus, the space complexity of
the representation is minimized. We have first considered a trivial case that can be represented
by the simplest Representative-Coefficient method, followed by some non-trivial cases where
we would require the Principal Component Analysis (PCA) method. We then discuss the
algorithm and limitations of PCA and see how some of these limitations can be solved using
the Kernel PCA method. We first examine the properties and construction of kernel functions
and then discuss the algorithm and limitations of KPCA. Next, to tackle the computational
inefficiency of KPCA, we introduce the Nystrom sampling approximation. This forms the base
of our randomized approach toward Kernel PCA and the study of the tradeoff involved in this
approach.
