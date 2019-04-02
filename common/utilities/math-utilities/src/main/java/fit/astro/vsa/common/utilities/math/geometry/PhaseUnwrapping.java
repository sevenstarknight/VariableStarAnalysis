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
package fit.astro.vsa.common.utilities.math.geometry;

import fit.astro.vsa.common.bindings.math.vector.RoundFunction;
import java.util.List;
import fit.astro.vsa.common.bindings.math.vector.FixFunction;
import fit.astro.vsa.common.utilities.math.relational.RelationalOperations;
import fit.astro.vsa.common.utilities.math.relational.RelationalOperators;
import fit.astro.vsa.common.bindings.math.vector.ModFunction;
import fit.astro.vsa.common.utilities.math.linearalgebra.VectorOperations;
import org.apache.commons.math3.analysis.function.Abs;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class PhaseUnwrapping {

    //Empty Constructor
    private PhaseUnwrapping(){
    }
    
    /**
     * unwraps radian phases P by changing absolute jumps greater than to pi to
     * their 2*pi complement. It unwraps along the first non-singleton dimension
     * of P and leaves the first phase value along this dimension unchanged. P
     *
     * @param p
     * @return
     */
    public static RealVector Unwrap(RealVector p) {
        double cutoff = Math.PI;

        return LocalUnwrap(p, cutoff);
    }

    /**
     *
     * @param p
     * @return
     */
    public static RealVector WrapTo2Pi(RealVector p) {
        List<Boolean> positiveInput = RelationalOperations
                .compareVector2Pt(p, 0.0, RelationalOperators.GT);

        p = p.map(new ModFunction(2.0 * Math.PI));

        List<Boolean> isZero = RelationalOperations
                .compareVector2Pt(p, 0, RelationalOperators.EQ);

        for (int idx = 0; idx < p.getDimension(); idx++) {
            if (positiveInput.get(idx) && isZero.get(idx)) {
                p.setEntry(idx, 2.0 * Math.PI);
            }
        }
        return p;
    }

    /**
     * Unwraps column vector of phase values.
     *
     * @param p
     * @param cutoff
     * @return
     */
    private static RealVector LocalUnwrap(RealVector p, double cutoff) {

        // Unwrap phase angles.  Algorithm minimizes the incremental phase variation 
        // by constraining it to the range [-pi,pi]
        RealVector dp = VectorOperations.diffArray(p);

        // Compute an integer describing how many times 2*pi we are off:
        // dp in [-pi, pi]: dp_corr = 0,
        // elseif dp in [-3*pi, 3*pi]: dp_corr = 1,
        // else if dp in [-5*pi, 5*pi]: dp_corr = 2, ...
        RealVector dp_corr = dp.mapDivide(2.0 * Math.PI);

        RealVector remAbs = dp_corr.map(new ModFunction(1)).map(new Abs());

        // ================================================================
        List<Boolean> roundDown = RelationalOperations.compareVector2Pt(
                remAbs, 0.5, RelationalOperators.LTEQ);

        dp_corr = RelationalOperations.applyRelationOperations(new FixFunction(), dp_corr, roundDown);

        // We want to do round(dp_corr), except that we want the tie-break at n+0.5
        // to round towards zero instead of away from zero (that is, (2n+1)*pi will
        // be shifted by 2n*pi, not by (2n+2)*pi):
        dp_corr = dp_corr.map(new RoundFunction());
        // ================================================================
        List<Boolean> stopJump = RelationalOperations.compareVector2Pt(
                dp.map(new Abs()), cutoff, RelationalOperators.LT);

        // Stop the jump from happening if dp < cutoff (no effect if cutoff <= pi)
        dp_corr = RelationalOperations.applyRelationOperations((double x) -> 0.0, dp_corr, roundDown);

        RealVector dpCumSum = VectorOperations.cumulativeSummationOfElements(dp_corr);

        RealVector tmpP = p.getSubVector(1, p.getDimension() - 1);

        // Integrate corrections and add to P to produce smoothed phase values
        p.setSubVector(1, tmpP.subtract(dpCumSum.mapMultiply(2.0 * Math.PI)));

        return p;
    }
}
