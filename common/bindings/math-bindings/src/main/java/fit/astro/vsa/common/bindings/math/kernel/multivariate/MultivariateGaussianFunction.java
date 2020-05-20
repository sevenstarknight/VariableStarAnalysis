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


package fit.astro.vsa.common.bindings.math.kernel.multivariate;

import fit.astro.vsa.common.bindings.math.kernel.KernelMethod;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * http://www.buch-kromann.dk/tine/nonpar/Nonparametric_Density_Estimation_multidim.pdf
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class MultivariateGaussianFunction implements KernelMethod, MultivariateKernelFunction {

    /**
     * Construct.
     */
    public MultivariateGaussianFunction() {
    }

    /**
     * Returns K_H.
     *
     * @param x
     * @param H
     * @return
     */
    @Override
    public double value(RealVector x, RealMatrix H) {

        LUDecomposition lud = new LUDecomposition(H);
        double det = lud.getDeterminant();
        RealMatrix invH = lud.getSolver().getInverse();

        double scale = Math.sqrt(Math.pow(2 * Math.PI, x.getDimension()) * det);

        double distance = x.dotProduct(invH.operate(x));

        return (1.0 / scale) * Math.exp(-0.5 * distance);

    }

}
