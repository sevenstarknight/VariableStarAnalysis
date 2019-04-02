/*
 * Copyright (C) 2018 Kyle Johnston <kyjohnst2000@my.fit.edu>
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
package fit.astro.vsa.utilities.ml.metriclearning.sj;

import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import java.util.Map;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class SJ_MetricLearningGradientGenerator {

    private final Map<Integer, RealVector> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;
    private final Map<String, Map<Integer, RealVector>> classMembers;

    // =======================================
    private double GAMMA = 0.5;

    public SJ_MetricLearningGradientGenerator(Map<Integer, RealVector> mapOfPatterns,
            Map<Integer, String> mapOfClasses,
            Map<String, Map<Integer, RealVector>> classMembers) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;
        this.classMembers = classMembers;
    }

    /**
     *
     * @param lk
     * <p>
     * @return gradiantOfFwrtL
     */
    public RealMatrix execute(RealMatrix lk) {
        RealMatrix mk = lk.transpose().multiply(lk);

        // =============================================
        RealMatrix reg = lk.multiply(mk).scalarMultiply(4*GAMMA);

        // ===============  Push Error ========================
        RealMatrix sumijl = MatrixUtils.createRealMatrix(
                mk.getRowDimension(), mk.getColumnDimension());

        for (Integer idx : mapOfPatterns.keySet()) {

            RealVector x_i = mapOfPatterns.get(idx);
            String label_i = mapOfClasses.get(idx);

            // Find similar
            Map<Integer, RealVector> listxj = classMembers.get(label_i);

            // sum_ij
            for (Integer jdx : listxj.keySet()) {

                //Same class
                RealVector x_j = listxj.get(jdx);

                // sum_ik
                for (Integer kdx : mapOfPatterns.keySet()) {

                    // 1 - y_il
                    if (mapOfClasses.get(kdx)
                            .equalsIgnoreCase(mapOfClasses.get(idx))) {
                        continue;
                    }

                    RealVector x_l = mapOfPatterns.get(kdx);

                    RealVector deltaij = x_i.subtract(x_j);
                    RealVector deltail = x_i.subtract(x_l);
                    
                    RealMatrix cij = deltaij.outerProduct(deltaij);
                    RealMatrix cil = deltail.outerProduct(deltail);
                    
                    double z = 1 + mk.multiply(cij).getTrace() - mk.multiply(cil).getTrace();
                    
                    double prime = SupportingFunctionality.HingePrimeApproxGLL(z);
                    
                    
                    sumijl = sumijl.add(cil.subtract(cij).scalarMultiply(prime));
                }
            }
        }
        
        return reg.add(lk.multiply(sumijl).scalarMultiply(2*(1-GAMMA)));
    }

    /**
     * 
     * @param GAMMA 
     */
    public void setGAMMA(double GAMMA) {
        this.GAMMA = GAMMA;
    }

    
}
