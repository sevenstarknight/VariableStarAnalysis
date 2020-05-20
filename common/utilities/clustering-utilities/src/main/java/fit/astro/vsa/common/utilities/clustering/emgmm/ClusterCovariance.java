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
package fit.astro.vsa.common.utilities.clustering.emgmm;

import java.util.Map;
import org.apache.commons.math3.analysis.function.Power;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class ClusterCovariance {

    private final Map<Integer, RealVector> setOfTrainingData;
    private final double probSum;
    private final Map<Integer, Double> expectationSet;
    private final RealVector meanCenter;

    public ClusterCovariance(
            Map<Integer, RealVector> setOfTrainingData,
            double probSum, Map<Integer, Double> expectationSet,
            RealVector meanCenter) {
        this.setOfTrainingData = setOfTrainingData;
        this.probSum = probSum;
        this.expectationSet = expectationSet;
        this.meanCenter = meanCenter;
    }

    public RealMatrix estimate(CovarianceType covType) {
        RealMatrix covMatrix;
        switch (covType) {
            case HeteroscedasticUnrestricted:
                covMatrix = HeteroscedasticUnrestricted();
                break;
            case HeteroscedasticDiagonal:
                covMatrix = HeteroscedasticDiagonal();
                break;
            case HeteroscedasticIsotropic:
                covMatrix = HeteroscedasticIsotropic();
                break;
            default:
                covMatrix = HeteroscedasticUnrestricted();
        }

        return covMatrix;
    }

    private RealMatrix HeteroscedasticUnrestricted() {

        RealMatrix summation = MatrixUtils.createRealMatrix(
                meanCenter.getDimension(), meanCenter.getDimension());

        for (Integer idx : setOfTrainingData.keySet()) {
            RealVector current = setOfTrainingData.get(idx);
            RealVector delta = current.subtract(meanCenter);

            summation = summation.add(
                    delta.outerProduct(delta).scalarMultiply(expectationSet.get(idx)));
        }

        return summation.scalarMultiply(1.0 / probSum);
    }

    private RealMatrix HeteroscedasticDiagonal() {

        RealVector summation = new ArrayRealVector(meanCenter.getDimension());

        for (Integer idx : setOfTrainingData.keySet()) {
            RealVector current = setOfTrainingData.get(idx);
            RealVector delta = current.subtract(meanCenter);

            summation = summation.add(
                    delta.map(new Power(2)).mapMultiply(expectationSet.get(idx)));
        }

        summation = summation.mapMultiply(1.0 / probSum);
        return MatrixUtils.createRealDiagonalMatrix(summation.toArray());
    }

    private RealMatrix HeteroscedasticIsotropic() {

        double summation = 0;

        for (Integer idx : setOfTrainingData.keySet()) {
            RealVector current = setOfTrainingData.get(idx);
            RealVector delta = current.subtract(meanCenter);

            summation += delta.getNorm() * expectationSet.get(idx);
        }

        summation = summation * (1.0 / (meanCenter.getDimension() * probSum));

        return MatrixUtils.createRealIdentityMatrix(
                meanCenter.getDimension()).scalarMultiply(summation);
    }

}
