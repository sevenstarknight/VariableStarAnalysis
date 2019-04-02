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
package fit.astro.vsa.common.bindings.math.geometry;

import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ConeEquation {

    private final RealVector t;
    private final RealVector b;
    private final Angle aperture;

    /**
     * @param t coordinates of apex point of cone
     * @param b coordinates of center of basement circle
     * @param aperture in radians
     */
    public ConeEquation(RealVector t, RealVector b, Angle aperture) {
        this.t = t;
        this.b = b;
        this.aperture = aperture;
    }

    public RealVector getT() {
        return t;
    }

    public RealVector getB() {
        return b;
    }

    public Angle getAperture() {
        return aperture;
    }

    /**
     * A right cone of height h and base radius r oriented along the z-axis,
     * with vertex pointing up, and with the base located at z=h can be
     * described by the parametric equations
     *
     * http://mathworld.wolfram.com/Cone.html
     *
     * cone along z-axis, in coordinates of origin and normal
     * 
     * orient along 
     * http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
     *
     * @param u [0,h] h = ||t - b||
     * @param theta [0, 2pi]
     * @return
     */
    public RealVector generatePoint(double u, Angle theta) {

        double h = t.subtract(b).getNorm();
        double r = aperture.tanHalfAngle() * (t.subtract(b).getNorm());

        double x = t.getEntry(0) + ((h - u) / h) * r * theta.cos();
        double y = t.getEntry(1) + ((h - u) / h) * r * theta.sin();
        double z = t.getEntry(2) + (h - u);

        //def in z
        RealVector coneAxisUnit = MatrixUtils.createRealVector(new double[]{0, 0, 1});

        Vector3D a = new Vector3D(coneAxisUnit.toArray());
        Vector3D c = new Vector3D(b.subtract(t).toArray());

        Vector3D v = Vector3D.crossProduct(a, c);

        double cosineAngle = a.dotProduct(c);

        RealMatrix skewSymmetric = MatrixUtils.createRealMatrix(new double[][]{
            {0.0, -v.getZ(), v.getY()}, {v.getZ(), 0.0, -v.getX()}, {-v.getY(), v.getX(), 0.0}
        });

        RealMatrix rotMatrix = MatrixUtils.createRealIdentityMatrix(3)
                .add(skewSymmetric)
                .add(skewSymmetric.multiply(skewSymmetric).scalarMultiply(1 / (1 + cosineAngle)));

        RealVector pointAlong = MatrixUtils.createRealVector(new double[]{x, y, z});

        rotMatrix.operate(pointAlong);

        return rotMatrix.operate(pointAlong);

    }

}
