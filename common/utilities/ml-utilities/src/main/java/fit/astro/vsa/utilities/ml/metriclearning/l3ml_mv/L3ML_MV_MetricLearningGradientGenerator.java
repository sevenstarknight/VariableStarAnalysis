/*
 * Copyright (C) 2018 Kyle Johnston 
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without isEven the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.utilities.ml.metriclearning.l3ml_mv;

import fit.astro.vsa.utilities.ml.MetricDistance_MV;
import fit.astro.vsa.utilities.ml.utils.SupportingFunctionality;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;

/**
 *
 * @author Kyle Johnston
 */
public class L3ML_MV_MetricLearningGradientGenerator {

    //============================================================ 
    // Input
    private final Map<Integer, List<Integer>> classMemberNear;
    private final Map<Integer, Map<String, RealMatrix>> mapOfPatterns;
    private final Map<Integer, String> mapOfClasses;

    private final L3ML_MV_MetricLearningObjective l3mlObj;

    //============================================================ 
    // Internal
    private RealMatrix sumij_g;
    private RealMatrix sumij_n;

    private RealMatrix sumijl_g;
    private RealMatrix sumijl_n;

    private RealMatrix sumijq_g;
    private RealMatrix sumijq_n;

    //============================================================ 
    private double LAMBDA = 0.5;
    private double GAMMA = 0.5;
    private double MU = 0.5;

    //============================================================    
    private double p = 2;

    /**
     *
     * @param classMemberNear
     * @param mapOfPatterns
     * @param mapOfClasses
     */
    public L3ML_MV_MetricLearningGradientGenerator(
            Map<Integer, List<Integer>> classMemberNear,
            Map<Integer, Map<String, RealMatrix>> mapOfPatterns,
            Map<Integer, String> mapOfClasses) {
        this.classMemberNear = classMemberNear;
        this.mapOfPatterns = mapOfPatterns;
        this.mapOfClasses = mapOfClasses;

        this.l3mlObj = new L3ML_MV_MetricLearningObjective(
                classMemberNear, mapOfPatterns, mapOfClasses);
    }

    /**
     *
     * @param kdx
     * @param l3mlVariables
     * @return gradJ
     */
    public RealMatrix[] generateLk(String kdx, Map<String, L3MLVariable_MV> l3mlVariables) {

        L3MLVariable_MV l3mlVariable = l3mlVariables.get(kdx);

        RealMatrix gammak = l3mlVariable.getGammak();
        RealMatrix nuk = l3mlVariable.getNuk();

        RealMatrix uk = l3mlVariable.getUk();
        RealMatrix vk = l3mlVariable.getVk();

        MetricDistance_MV metricDistanceK = new MetricDistance_MV(uk, vk);

        double scaleMN = l3mlVariable.getUk().getColumnDimension()*l3mlVariable.getVk().getColumnDimension();
        
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

            RealMatrix x_i = mapOfPatterns.get(idx).get(kdx);
            abijs.put(idx, new ArrayList<>(classMemberNear.get(idx).size()));

            classMemberNear.get(idx).parallelStream().map((jdx)
                    -> mapOfPatterns.get(jdx).get(kdx)).map((x_j)
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

        sumij_g = sumij_g.scalarMultiply(1.0/scaleMN);
        sumij_n = sumij_n.scalarMultiply(1.0/scaleMN);
        
        // ================================================
        // Push Terms
        sumijl_g = MatrixUtils.createRealMatrix(
                gammak.getRowDimension(), gammak.getColumnDimension());

        sumijl_n = MatrixUtils.createRealMatrix(
                nuk.getRowDimension(), nuk.getColumnDimension());

        for (Integer idx : classMemberNear.keySet()) {

            RealMatrix x_i = mapOfPatterns.get(idx).get(kdx);
            // pre generated neighbors
            List<Pair<RealMatrix, RealMatrix>> pairAB = abijs.get(idx);

            for (Pair<RealMatrix, RealMatrix> pairab : pairAB) {

                mapOfPatterns.keySet().parallelStream().filter((ldx) -> !(mapOfClasses.get(ldx)
                        .equalsIgnoreCase(mapOfClasses.get(idx)))).map((ldx)
                        -> mapOfPatterns.get(ldx).get(kdx)).forEachOrdered((x_l) -> {
                    RealMatrix deltail = x_i.subtract(x_l);

                    RealMatrix delA_ijl = pairab.getFirst()
                            .subtract(makeAij(deltail, vk));

                    RealMatrix delB_ijl = pairab.getSecond()
                            .subtract(makeBij(deltail, uk));

                    double distanceij = (uk.multiply(pairab.getFirst())).getTrace();

                    double z = 1 + distanceij
                            - metricDistanceK.matrixDistance(x_i, x_l);

                    // εi,j,lm=1−( xi−xlm)TLTL( xi−xlm)+( xi− xj )TLTL( xi− xj ),
                    // and if εijlm > 0, [εijlm]+ = 1, otherwise, [εijlm]+ = 0.
                    double hPrime = SupportingFunctionality.HingePrimeApproxGLL(z);

                    sumijl_g = sumijl_g.add(delA_ijl.scalarMultiply(hPrime));

                    sumijl_n = sumijl_n.add(delB_ijl.scalarMultiply(hPrime));
                });
            }
        }

        sumijl_g = sumijl_g.scalarMultiply(1.0/scaleMN);
        sumijl_n = sumijl_n.scalarMultiply(1.0/scaleMN);
        
        // ==========================================================
        RealMatrix regGamma = MatrixUtils.createRealIdentityMatrix(
                gammak.getColumnDimension()).scalarMultiply(LAMBDA);
        
        regGamma = regGamma.scalarMultiply(1.0 / (gammak.getColumnDimension()*gammak.getColumnDimension()));
        
        RealMatrix regNu = MatrixUtils.createRealIdentityMatrix(
                nuk.getColumnDimension()).scalarMultiply(LAMBDA);
        
        regNu = regNu.scalarMultiply(1.0 / (nuk.getColumnDimension()*nuk.getColumnDimension()));

        // ==========================================================
        // Complete Grad for ik Metric
        RealMatrix[] gradOut_ik = new RealMatrix[2];
        gradOut_ik[0] = gammak.multiply(
                (sumij_g.scalarMultiply(1 - GAMMA)).add(sumijl_g.scalarMultiply(GAMMA)).add(regGamma)).scalarAdd(2.0);
        gradOut_ik[1] = nuk.multiply(
                (sumij_n.scalarMultiply(1 - GAMMA)).add(sumijl_n.scalarMultiply(GAMMA)).add(regNu)).scalarAdd(2.0);

        gradOut_ik[0] = gradOut_ik[0].scalarMultiply(Math.pow(l3mlVariable.getWeight(), p));
        gradOut_ik[1] = gradOut_ik[1].scalarMultiply(Math.pow(l3mlVariable.getWeight(), p));

        // ==========================================================
        sumijq_g = MatrixUtils.createRealMatrix(
                gammak.getRowDimension(), gammak.getColumnDimension());

        sumijq_n = MatrixUtils.createRealMatrix(
                nuk.getRowDimension(), nuk.getColumnDimension());

        
        for (String qdx : l3mlVariables.keySet()) {

            if (qdx.contentEquals(kdx)) {
                continue;
            }

            L3MLVariable_MV l3mlVariableQ = l3mlVariables.get(qdx);


            RealMatrix uq = l3mlVariableQ.getUk();
            RealMatrix vq = l3mlVariableQ.getVk();

            MetricDistance_MV metricDistanceQ = new MetricDistance_MV(uq, vq);

            double scaleAB = l3mlVariableQ.getUk().getColumnDimension()*l3mlVariableQ.getVk().getColumnDimension();
            
            for (Integer idx : mapOfPatterns.keySet()) {
                RealMatrix x_i_q = mapOfPatterns.get(idx).get(qdx);
                RealMatrix x_i_k = mapOfPatterns.get(idx).get(kdx);

                List<Integer> listxj = classMemberNear.get(idx);

                listxj.parallelStream().filter((jdx) -> !(Objects.equals(idx, jdx))).forEachOrdered((jdx) -> {
                    RealMatrix x_j_q = mapOfPatterns.get(jdx).get(qdx);
                    RealMatrix x_j_k = mapOfPatterns.get(jdx).get(kdx);

                    RealMatrix deltaij_k = x_i_k.subtract(x_j_k);
                    RealMatrix aij_k = makeAij(deltaij_k, vk)
                            .scalarMultiply(1.0/(uk.getColumnDimension()*vk.getColumnDimension()));
                    RealMatrix bij_k = makeBij(deltaij_k, uk)
                            .scalarMultiply(1.0/(uk.getColumnDimension()*vk.getColumnDimension()));

                    double d_ij_q = metricDistanceQ.matrixDistance(x_i_q, x_j_q)
                            /(uq.getColumnDimension()*vq.getColumnDimension())/scaleAB;
                    double d_ij_k = metricDistanceK.matrixDistance(x_i_k, x_j_k)
                            /(uk.getColumnDimension()*vk.getColumnDimension())/scaleMN;

                    sumijq_g = sumijq_g.add(aij_k.scalarMultiply((d_ij_k - d_ij_q)));
                    sumijq_n = sumijq_n.add(bij_k.scalarMultiply((d_ij_k - d_ij_q)));
                });
            }

        }

        sumijq_g = gammak.multiply(sumijq_g).scalarMultiply(4.0 * MU).scalarMultiply(1.0/scaleMN);
        sumijq_n = nuk.multiply(sumijq_n).scalarMultiply(4.0 * MU).scalarMultiply(1.0/scaleMN);

        RealMatrix[] gradOut_jk = new RealMatrix[2];
        gradOut_jk[0] = gradOut_ik[0].add(sumijq_g);
        gradOut_jk[1] = gradOut_ik[1].add(sumijq_n);

        return gradOut_jk;
    }

    /**
     *
     * @param l3mlVariables
     * @return 
     */
    public Map<String, Double> updateWeight(
            Map<String, L3MLVariable_MV> l3mlVariables) {

        Map<String, Double> ikNumMap = new HashMap<>(l3mlVariables.keySet().size());
        Map<String, Double> ik = new HashMap<>(l3mlVariables.keySet().size());
        
        double sumOverK = 0;
        for (String ldx : l3mlVariables.keySet()) {
            ik.put(ldx, l3mlObj.valueIK(ldx, l3mlVariables));
            
            double lkNum = Math.pow(1.0 / ik.get(ldx), 1.0 / (1.0 - p));

            ikNumMap.put(ldx, lkNum);

            sumOverK = sumOverK + lkNum;
        }

        for (String ldx : l3mlVariables.keySet()) {
            l3mlVariables.get(ldx).setWeight(ikNumMap.get(ldx) / sumOverK);
        }

        return ik;
    }

    private RealMatrix makeAij(RealMatrix deltaij, RealMatrix vk) {
        return deltaij.transpose().multiply(vk).multiply(deltaij);
    }

    private RealMatrix makeBij(RealMatrix deltaij, RealMatrix uk) {
        return deltaij.multiply(uk).multiply(deltaij.transpose());
    }

    /**
     * Regularization Parameter
     *
     * @param LAMBDA
     */
    public void setLAMBDA(double LAMBDA) {
        this.LAMBDA = LAMBDA;
    }

    /**
     * Push/Pull Parameter
     *
     * @param GAMMA
     */
    public void setGAMMA(double GAMMA) {
        this.GAMMA = GAMMA;
    }

    /**
     * Cross Parameter
     *
     * @param MU
     */
    public void setMU(double MU) {
        this.MU = MU;
    }

    /**
     *
     * @param p
     */
    public void setP(double p) {
        this.p = p;
    }

}
