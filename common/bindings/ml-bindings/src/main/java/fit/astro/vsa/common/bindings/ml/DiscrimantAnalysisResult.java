/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.bindings.ml;

import java.util.Map;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Output from the discriminate classifiers
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class DiscrimantAnalysisResult {

//<editor-fold defaultstate="collapsed" desc="Variables">
    private final Map<Integer, RealVector> reducedSet;
    private final RealMatrix covarianceMatrix;

    private final RealVector meanCentroid;

    private final double nSize;

    private final RealMatrix invCovMatrix;
    private final double logDeterminant;
    private final double constant;
//</editor-fold>

    /**
     * Output from the discriminate classifiers
     *
     * @param reducedSet the reduced data set (subgroup)
     * @param meanCentroid The mean centroid for the subgroup
     * @param covarianceMatrix The covariance matrix for the subgroup
     * @param invCovMatrix The inverse covariance matrix for the subgroup
     * @param nTotal the total number in the subgroup
     * @param logDeterminant the log of the determinate
     */
    public DiscrimantAnalysisResult(
            Map<Integer, RealVector> reducedSet,
            RealVector meanCentroid,
            RealMatrix covarianceMatrix, RealMatrix invCovMatrix,
            double logDeterminant,
            double nTotal) {
        this.reducedSet = reducedSet;

        this.covarianceMatrix = covarianceMatrix;

        this.nSize = reducedSet.size();

        this.meanCentroid = meanCentroid;

        this.invCovMatrix = invCovMatrix;

        // General QDA form
        this.logDeterminant = Math.log(
                new LUDecomposition(covarianceMatrix).getDeterminant());

        this.constant = Math.log(nSize / nTotal) - 0.5 * logDeterminant;
    }

//<editor-fold defaultstate="collapsed" desc="Getters and Setters">
    /**
     * @return the reducedSet
     */
    public Map<Integer, RealVector> getReducedSet() {
        return reducedSet;
    }

    /**
     * @return the covarianceMatrix
     */
    public RealMatrix getCovarianceMatrix() {
        return covarianceMatrix;
    }

    /**
     * @return the meanCentroid
     */
    public RealVector getMeanCentroid() {
        return meanCentroid;
    }

    /**
     * @return the nSize
     */
    public double getnSize() {
        return nSize;
    }

    /**
     * @return the invCovMatrix
     */
    public RealMatrix getInvCovMatrix() {
        return invCovMatrix;
    }

    /**
     * @return the logDeterminant
     */
    public double getLogDeterminant() {
        return logDeterminant;
    }

    /**
     * @return the constant
     */
    public double getConstant() {
        return constant;
    }
//</editor-fold>
}
