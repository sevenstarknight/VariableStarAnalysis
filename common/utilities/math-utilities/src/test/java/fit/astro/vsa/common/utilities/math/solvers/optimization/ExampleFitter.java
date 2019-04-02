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
package fit.astro.vsa.common.utilities.math.solvers.optimization;

import java.util.Collection;
import org.apache.commons.math3.fitting.AbstractCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ExampleFitter extends AbstractCurveFitter {

    private final RealVector coefficientsGuess;

    /**
     * Levenberg Marquardt
     * <p>
     * @param coefficientsGuess
     */
    public ExampleFitter(
            RealVector coefficientsGuess) {
        this.coefficientsGuess = coefficientsGuess;
    }

    @Override
    protected LeastSquaresProblem getProblem(
            Collection<WeightedObservedPoint> points) {

        int len = points.size();
        double[] target = new double[len];
        double[] weights = new double[len];

        int idx = 0;
        for (WeightedObservedPoint point : points) {
            target[idx] = point.getY();
            weights[idx] = point.getWeight();
            idx++;
        }

        AbstractCurveFitter.TheoreticalValuesFunction model
                = new AbstractCurveFitter.TheoreticalValuesFunction(
                        new ExampleModel(), points);

        return new LeastSquaresBuilder().
                maxEvaluations(Integer.MAX_VALUE).
                maxIterations(Integer.MAX_VALUE).
                start(coefficientsGuess).
                target(target).
                weight(new DiagonalMatrix(weights)).
                model(model.getModelFunction(),
                        model.getModelFunctionJacobian()).
                build();

    }

}
