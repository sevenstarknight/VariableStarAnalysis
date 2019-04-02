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
package fit.astro.vsa.utilities.ml.metriclearning.lmnn_mv;

import fit.astro.vsa.utilities.ml.MetricDistance_MV;
import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;

/**
 * Song, Kun, et al. "Parameter Free Large Margin Nearest Neighbor for Distance
 * Metric Learning." AAAI. 2017.
 *
 * @author Kyle Johnston
 */
public class LMNN_MV_MetricLearningGradientGenerator {

    private final Map<Integer, RealMatrix> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    //============================================================ 
    // Internal
    private RealMatrix sumij_g;
    private RealMatrix sumij_n;

    private RealMatrix sumijl_g;
    private RealMatrix sumijl_n;

    private RealMatrix sumijq_g;
    private RealMatrix sumijq_n;

    // =======================================
    private final Map<Integer, List<Integer>> classMemberNear;

    private double GAMMA = 0.5;
    private double LAMBDA = 0.5;

    /**
     *
     * @param classMemberNear
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public LMNN_MV_MetricLearningGradientGenerator(
            Map<Integer, List<Integer>> classMemberNear,
            Map<Integer, RealMatrix> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;

        // ============= Neighbors =====================
        this.classMemberNear = classMemberNear;

    }

    /**
     *
     * @param gammak
     * @param nuk
     * <p>
     * @return gradiantOfFwrtL
     */
    public RealMatrix[] execute(RealMatrix gammak, RealMatrix nuk) {

        RealMatrix uk = (gammak.transpose()).multiply(gammak);
        RealMatrix vk = (nuk.transpose()).multiply(nuk);

        MetricDistance_MV metricDistanceK = new MetricDistance_MV(uk, vk);

        // =============================================
        // Pull Terms
        sumij_g = MatrixUtils.createRealMatrix(
                gammak.getRowDimension(), gammak.getColumnDimension());

        sumij_n = MatrixUtils.createRealMatrix(
                nuk.getRowDimension(), nuk.getColumnDimension());

        // In neighborhood
        Map<Integer, List<Pair<RealMatrix, RealMatrix>>> abijs
                = new HashMap<>(classMemberNear.size());

        // =============================================
        // generate aij, bij, and sums
        for (Integer idx : classMemberNear.keySet()) {

            RealMatrix x_i = mapOfPatterns.get(idx);
            abijs.put(idx, new ArrayList<>(classMemberNear.get(idx).size()));

            classMemberNear.get(idx).parallelStream().map((jdx)
                    -> mapOfPatterns.get(jdx)).map((x_j)
                    -> x_i.subtract(x_j)).forEachOrdered((deltaij) -> {
                // =============================================
                // sum_ij for gamma
                RealMatrix aij = makeAij(deltaij, vk);
                sumij_g = sumij_g.add(aij);

                // =============================================
                // sum_ij for nu
                RealMatrix bij = makeBij(deltaij, uk);
                sumij_n = sumij_n.add(bij);

                abijs.get(idx).add(new Pair<>(aij, bij));
            });
        }

        // ================================================
        // Push Terms
        sumijl_g = MatrixUtils.createRealMatrix(
                gammak.getRowDimension(), gammak.getColumnDimension());

        sumijl_n = MatrixUtils.createRealMatrix(
                nuk.getRowDimension(), nuk.getColumnDimension());

        for (Integer idx : classMemberNear.keySet()) {

            RealMatrix x_i = mapOfPatterns.get(idx);
            // pre generated neighbors
            List<Pair<RealMatrix, RealMatrix>> pairAB = abijs.get(idx);

            for (Pair<RealMatrix, RealMatrix> pairab : pairAB) {

                mapOfPatterns.keySet().parallelStream().filter((ldx) -> !(mapOfClasses.get(ldx)
                        .equalsIgnoreCase(mapOfClasses.get(idx)))).map((ldx)
                        -> mapOfPatterns.get(ldx)).forEachOrdered((x_l) -> {
                    RealMatrix deltail = x_i.subtract(x_l);

                    RealMatrix delA = pairab.getFirst()
                            .subtract(makeAij(deltail, vk));

                    RealMatrix delB = pairab.getSecond()
                            .subtract(makeBij(deltail, uk));

                    double distanceij = (uk.multiply(pairab.getFirst())).getTrace();

                    double z = 1 + distanceij
                            - metricDistanceK.matrixDistance(x_i, x_l);

                    // εi,j,lm=1−( xi−xlm)TLTL( xi−xlm)+( xi− xj )TLTL( xi− xj ),
                    // and if εijlm > 0, [εijlm]+ = 1, otherwise, [εijlm]+ = 0.
                    double hPrime = SupportingFunctionality.HingePrimeApproxGLL(z);

                    sumijl_g = sumijl_g.add(delA.scalarMultiply(hPrime));

                    sumijl_n = sumijl_n.add(delB.scalarMultiply(hPrime));
                });
            }
        }

        RealMatrix regGamma = MatrixUtils.createRealIdentityMatrix(
                gammak.getColumnDimension()).scalarMultiply(LAMBDA);
        RealMatrix regNu = MatrixUtils.createRealIdentityMatrix(
                nuk.getColumnDimension()).scalarMultiply(LAMBDA);

        // Complete Grad for Metric
        RealMatrix[] gradOut_ik = new RealMatrix[2];
        gradOut_ik[0] = gammak.multiply(
                (sumij_g.scalarMultiply(1 - GAMMA)).add(sumijl_g.scalarMultiply(GAMMA)).add(regGamma)).scalarAdd(2.0);
        gradOut_ik[1] = nuk.multiply(
                (sumij_n.scalarMultiply(1 - GAMMA)).add(sumijl_n.scalarMultiply(GAMMA)).add(regNu)).scalarAdd(2.0);

        return gradOut_ik;
    }

    private RealMatrix makeAij(RealMatrix deltaij, RealMatrix vk) {
        return deltaij.transpose().multiply(vk).multiply(deltaij);
    }

    private RealMatrix makeBij(RealMatrix deltaij, RealMatrix uk) {
        return deltaij.multiply(uk).multiply(deltaij.transpose());
    }

    public void setLAMBDA(double LAMBDA) {
        this.LAMBDA = LAMBDA;
    }

    public void setGAMMA(double GAMMA) {
        this.GAMMA = GAMMA;
    }

}
