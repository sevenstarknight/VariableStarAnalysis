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

import java.util.Random;
import fit.astro.vsa.common.bindings.math.geometry.Angle;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

/**
 * Sphere and Circle Point Picking:
 * http://mathworld.wolfram.com/SpherePointPicking.html
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class RandomSurfaceGenerator {

    // Empty Constructor
    private RandomSurfaceGenerator() {
    }

    /**
     * Random radian (0, 2pi]
     *
     * @return
     */
    public static RealVector getPointOnUnitCircle() {
        return getPointOnUnitCircle(new Random());
    }

    /**
     * Random radian (0, 2pi]
     *
     * @param random
     * @return
     */
    public static RealVector getPointOnUnitCircle(Random random) {
        // Put on [0, 1]
        Angle angle = Angle.fromRadians(
                random.nextDouble() * Math.PI * 2.0);

        RealVector xyPair = MatrixUtils.createRealVector(new double[]{
            angle.cos(),
            angle.sin()
        });

        return xyPair;

    }

    /**
     * Marsaglia, G. "Choosing a Point from the Surface of a Sphere." Ann. Math.
     * Stat. 43, 645-646, 1972.
     *
     * @return
     */
    public static RealVector getPointOnUnitSphere() {
        return getPointOnUnitSphere(new Random());
    }

    /**
     * Marsaglia, G. "Choosing a Point from the Surface of a Sphere." Ann. Math.
     * Stat. 43, 645-646, 1972.
     *
     * @param random
     * @return
     */
    public static RealVector getPointOnUnitSphere(Random random) {
        // Put on [-1, 1]
        double x1 = random.nextDouble() * 2 - 1;
        double x2 = random.nextDouble() * 2 - 1;

        // reject point if (x1 * x1 + x2 * x2) >= 1
        while ((x1 * x1 + x2 * x2) >= 1) {
            x1 = random.nextDouble() * 2 - 1;
            x2 = random.nextDouble() * 2 - 1;
        }

        // have a uniform distribution on the surface of a unit sphere
        return MatrixUtils.createRealVector(new double[]{
            2 * x1 * Math.sqrt(1 - x1 * x1 - x2 * x2),
            2 * x2 * Math.sqrt(1 - x1 * x1 - x2 * x2),
            (1 - 2 * (x1 * x1 + x2 * x2))});

    }

}
