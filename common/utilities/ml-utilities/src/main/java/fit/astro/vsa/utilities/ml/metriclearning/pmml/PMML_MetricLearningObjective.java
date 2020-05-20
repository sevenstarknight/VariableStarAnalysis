/*
 * Copyright (C) 2019 kjohnston
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
package fit.astro.vsa.utilities.ml.metriclearning.pmml;

import fit.astro.vsa.common.bindings.ml.metric.MultiViewMetric;
import java.util.Map;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author kjohnston
 */
public class PMML_MetricLearningObjective {

    private static final Logger LOGGER
            = LoggerFactory.getLogger(PMML_MetricLearningObjective.class);
    
    private final Map<int[], Double> del_ijs;

    private double sumLogDet;
    private double sumij;

    // 
    private double rho = 0.5;
    private double tau = 0.5;
    private double gamma = 0.5;

    /**
     *
     * @param del_ijs
     */
    public PMML_MetricLearningObjective(
            Map<int[], Double> del_ijs) {
        this.del_ijs = del_ijs;
    }

    public double estimateObjective(
            Map<int[], Double> eta_ijs,
            Map<String, MultiViewMetric> pmmlVariables) {

        //======================================================
        // Log Det Component (Regularization)
        sumLogDet = 0;
        for (String view : pmmlVariables.keySet()) {
            
            // 
            RealMatrix mk = pmmlVariables.get(view).getMk();

            RealMatrix mk_0 = MatrixUtils.createRealIdentityMatrix(
                    mk.getRowDimension()).scalarMultiply(1e0);

            
//            sumLogDet += (mk.subtract(mk_0)).getFrobeniusNorm();
            
            //
            RealMatrix mk_0_inv = new LUDecomposition(mk_0).getSolver().getInverse();

            RealMatrix mTimesm0 = mk.multiply(mk_0_inv);
            
            double trace = (mTimesm0).getTrace();

//            CholeskyDecomposition choDecomp = new CholeskyDecomposition(mTimesm0);
//            
//            RealMatrix lDecomp = choDecomp.getL();
//            double logDet = 0.0;
//            for(int idx = 0; idx < lDecomp.getColumnDimension(); idx++){
//                logDet += Math.log(lDecomp.getEntry(idx, idx));
//            }
//            logDet = logDet*2;
            
            double det = new LUDecomposition(mTimesm0).getDeterminant();
            double logDet = Math.log(det);

            sumLogDet += trace - logDet - mk.getRowDimension();
        }

        double firstElement = sumLogDet / (double) pmmlVariables.keySet().size();
        
        //======================================================
        // Margin Constraint
        sumij = 0;
        for (int[] idx : del_ijs.keySet()) {

            double eta = eta_ijs.get(idx);
            double del = del_ijs.get(idx);

            sumij += lossFunction(eta, del * rho - tau);
        }
        
        return firstElement + (gamma / (double) del_ijs.size()) * sumij;

    }

    private double lossFunction(double x, double x_0) {

        if (x <= x_0) {
            return 0.0;
        } else {
            return (x - x_0) * (x - x_0);
        }
    }

    public void setRho(double rho) {
        this.rho = rho;
    }

    public void setTau(double tau) {
        this.tau = tau;
    }

    public void setGamma(double gamma) {
        this.gamma = gamma;
    }

}
