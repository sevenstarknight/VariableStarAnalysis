/*
 * Copyright (C) 2018 Kyle Johnston 
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
package fit.astro.vsa.utilities.ml.metriclearning;

import fit.astro.vsa.common.bindings.math.matrix.UnivariateFunctionMapper;
import fit.astro.vsa.common.bindings.math.vector.MaxFunction;
import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston 
 */
public class PushPullMatrixAnalysis {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(PushPullMatrixAnalysis.class);
    
    private final Map<Integer, RealMatrix> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private final int[] sizeMatrix;

    private double GAMMMA = 0.5;
    private double LAMBDA = 0.5;

    /**
     * 
     * @param mapOfPatterns
     * @param mapOfClasses 
     */
    public PushPullMatrixAnalysis(
            Map<Integer, RealMatrix> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfClasses = mapOfClasses;
        this.mapOfPatterns = mapOfPatterns;

        RealMatrix example = mapOfPatterns.values().iterator().next();
        this.sizeMatrix = new int[]{example.getRowDimension(),
            example.getColumnDimension()};

    }

    /**
     * 
     * @return 
     */
    public RealMatrix execute() {

        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        Map<String, List<RealMatrix>> classMembers = LabelHandling
                .sortIntoMatrixClasses(mapOfPatterns, mapOfClasses);

        //  ================ Pull Terms =============================
        RealMatrix totalPull = MatrixUtils.createRealMatrix(
                sizeMatrix[0], sizeMatrix[0]);
        for (String c : classMembers.keySet()) {

            for (RealMatrix x_i : classMembers.get(c)) {
                for (RealMatrix x_j : classMembers.get(c)) {
                    RealMatrix deltaMatrix = x_i.subtract(x_j);
                    totalPull = totalPull.add(
                            deltaMatrix.multiply(deltaMatrix.transpose()));
                }
            }
            int nc = classMembers.get(c).size();
            totalPull = totalPull.scalarMultiply(1.0 / (nc - 1));
        }
        totalPull = totalPull.scalarMultiply(1.0 / (GAMMMA));

        //  ================ Push Terms =============================
        RealMatrix totalPush = MatrixUtils.createRealMatrix(
                sizeMatrix[0], sizeMatrix[0]);

        for (String c : classMembers.keySet()) {
            RealMatrix total = MatrixUtils.createRealMatrix(
                    sizeMatrix[0], sizeMatrix[0]);

            for (RealMatrix x_i : classMembers.get(c)) {
                for (String d : classMembers.keySet()) {
                    if (!c.equalsIgnoreCase(d)) {
                        for (RealMatrix x_j : classMembers.get(d)) {
                            RealMatrix deltaMatrix = x_i.subtract(x_j);
                            total = total.add(
                                    deltaMatrix.multiply(deltaMatrix.transpose()));
                        }
                    }
                }
            }
            totalPush = totalPush.add(total.scalarMultiply(
                    (1.0 / (mapOfPatterns.size() - classMembers.get(c).size()))));
        }

        totalPush = totalPush.scalarMultiply(1.0 / (GAMMMA));

        RealMatrix metric = (totalPush.scalarMultiply(LAMBDA))
                .subtract(totalPull.scalarMultiply(1.0 - LAMBDA));
        EigenDecomposition eigenDecomposition = new EigenDecomposition(metric);

        RealMatrix dMatrix = eigenDecomposition.getD();
        RealMatrix vMatrix = eigenDecomposition.getV();

        dMatrix.walkInOptimizedOrder(new UnivariateFunctionMapper(
                new MaxFunction(0.0)));

        metric = vMatrix.multiply(dMatrix).multiply(vMatrix.transpose());

        return metric;
    }

    public int[] getSizeMatrix() {
        return sizeMatrix;
    }

    
    
    /**
     * 
     * @param GAMMMA 
     */
    public void setGAMMMA(double GAMMMA) {
        this.GAMMMA = GAMMMA;
    }

    /**
     * 
     * @param LAMBDA 
     */
    public void setLAMBDA(double LAMBDA) {
        this.LAMBDA = LAMBDA;
    }

}
