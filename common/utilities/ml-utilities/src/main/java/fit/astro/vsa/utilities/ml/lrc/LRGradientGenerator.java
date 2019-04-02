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
package fit.astro.vsa.utilities.ml.lrc;


import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import java.util.Map;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class LRGradientGenerator {

    private final int number;
    private final int betaSize;

    private final RealMatrix delLdelBeta;
    private final RealMatrix doubleDelLDelBeta;

    /**
     * 
     * @param xTilda
     * @param setOfPatterns
     * @param yResponse
     * @param betaParams 
     */
    public LRGradientGenerator(
            RealMatrix xTilda,
            Map<Integer, RealVector> setOfPatterns,
            Map<Integer, RealVector> yResponse,
            Map<String, RealVector> betaParams) {

        // N
        this.number = setOfPatterns.size();

        // K
        this.betaSize = betaParams.size();

        // ================================
        // y and p
        RealMatrix yBold = MatrixUtils.createRealMatrix(number, betaSize);
        RealMatrix pBold = MatrixUtils.createRealMatrix(number, betaSize);

        int counter = 0;
        for (Integer idx : setOfPatterns.keySet()) {
            // Initialize
            RealVector currentPattern = setOfPatterns.get(idx);
            RealVector yResponseVector = yResponse.get(idx);

            // y-Matrix 
            yBold.setRowVector(counter, yResponseVector);

            // p-Matrix
            RealVector softMaxProb = SupportingFunctionality.SoftMax(
                    betaParams, currentPattern);

            pBold.setRowVector(counter, softMaxProb);

            counter++;
        }

        RealMatrix pBoldVector = MatrixUtils.createColumnRealMatrix(
                MatrixOperations.unpackMatrix(pBold));

        RealMatrix yBoldVector = MatrixUtils.createColumnRealMatrix(
                MatrixOperations.unpackMatrix(yBold));

        // ================================
        // Weights
        RealMatrix wBold = MatrixUtils.createRealMatrix(
                number * (betaSize), number * (betaSize));

        for (int idx = 0; idx < betaSize; idx++) {
            for (int jdx = 0; jdx < betaSize; jdx++) {
                RealMatrix w_km;
                if (idx == jdx) {
                    RealVector p_k = pBold.getColumnVector(idx);
                    RealVector ones = MatrixUtils.createRealVector(VectorOperations.ones(p_k.getDimension()));
                    w_km = MatrixUtils.createRealDiagonalMatrix(
                            p_k.ebeMultiply(ones.subtract(p_k)).toArray());
                } else {
                    RealVector p_k = pBold.getColumnVector(idx);
                    RealVector p_m = pBold.getColumnVector(jdx);
                    w_km = MatrixUtils.createRealDiagonalMatrix(
                            p_k.ebeMultiply(p_m).mapMultiply(-1.0).toArray());
                }
                wBold.setSubMatrix(w_km.getData(), idx * number, jdx * number);
            }
        }
        
        RealMatrix delta = yBoldVector.subtract(pBoldVector);

        delLdelBeta = xTilda.transpose().multiply(delta);

        doubleDelLDelBeta = xTilda.transpose().multiply(wBold).multiply(xTilda);

    }

    /**
     *
     * @return the gradient
     */
    public RealMatrix getDelLdelBeta() {
        return delLdelBeta;
    }

    /**
     *
     * @return the double grad
     */
    public RealMatrix getDoubleDelLDelBeta() {
        return doubleDelLDelBeta;
    }

}
