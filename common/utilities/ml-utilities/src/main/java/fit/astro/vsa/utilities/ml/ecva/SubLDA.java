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
package fit.astro.vsa.utilities.ml.ecva;

import fit.astro.vsa.common.utilities.math.linearalgebra.MatrixOperations;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.math3.analysis.function.Exp;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class SubLDA {

    private final Map<Integer, RealVector> posteriorMap;
    private final Map<Integer, String> classEstimateMap;

    /**
     * Linear Discriminant Analysis for ECVA function
     * <p>
     * @param xNew
     * @param xTrain
     * @param classMap
     * @param priorMap
     */
    public SubLDA(
            Map<Integer, RealVector> xNew,
            Map<Integer, RealVector> xTrain,
            Map<Integer, String> classMap,
            Map<String, Double> priorMap) {

        RealVector example = xNew.values().iterator().next();

        RealMatrix sWithin = MatrixUtils.createRealMatrix(
                example.getDimension(), example.getDimension());

        Map<String, Map<Integer, RealVector>> classMembers
                = LabelHandling.sortIntoMaps(xTrain, classMap);
        
        Map<String, RealVector> meanX = new HashMap<>(classMembers.keySet().size());

        // Initialize and calculate Swithin
        double counter = 0;
        String[] classLabels = classMembers.keySet().toArray(
                new String[classMembers.keySet().size()]);

        for (String e : classMembers.keySet()) {
            RealMatrix xSub = MatrixOperations.generateMatrixFromMap(classMembers.get(e));

            RealVector xMean = MatrixOperations.dimensionalMean(xSub, Boolean.TRUE);
            meanX.put(e, xMean);

            RealMatrix xm = xSub.subtract(MatrixOperations.replicateMatrixRows(xMean,
                    xSub.getRowDimension()));

            RealMatrix sTmp = xm.transpose().multiply(xm);

            sWithin = sWithin.add(sTmp);
            counter += xSub.getRowDimension();
        }

        sWithin = sWithin.scalarMultiply(1.0 / (double) (counter - classMembers.size()));

        Map<Integer, RealVector> listOfProbs = new HashMap<>(xNew.size());

        LUDecomposition lud = new LUDecomposition(sWithin);
        RealMatrix invWithin = lud.getSolver().getInverse();
        double detWith = lud.getDeterminant();

        xNew.keySet().forEach((idx) -> {
            RealVector lVector
                    = new ArrayRealVector(classMembers.keySet().size());
            int jdx = 0;
            for (String e : classMembers.keySet()) {
                RealVector xMean = meanX.get(e);
                Double priorProb = priorMap.get(e);

                RealVector deltaij = xNew.get(idx).subtract(xMean);

                lVector.setEntry(jdx, Math.log(priorProb)
                        - 0.05 * deltaij.dotProduct(invWithin.operate(deltaij))
                        + Math.log(detWith));
                jdx++;
            }

            listOfProbs.put(idx, lVector);
        });

        //==========================================================
        posteriorMap = new HashMap<>(listOfProbs.size());
        classEstimateMap = new HashMap<>(listOfProbs.size());

        listOfProbs.keySet().forEach((idx) -> {
            RealVector e = listOfProbs.get(idx);

            RealVector postProb = (e.mapSubtract(e.getMaxValue())).map(new Exp());
            posteriorMap.put(idx, postProb);

            classEstimateMap.put(idx, classLabels[postProb.getMaxIndex()]);
        });

    }

    /**
     * @return the posteriorMap
     */
    public Map<Integer, RealVector> getPosteriorMap() {
        return posteriorMap;
    }

    /**
     * @return the classEstimateMap
     */
    public Map<Integer, String> getClassEstimateMap() {
        return classEstimateMap;
    }

}
