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
package fit.astro.vsa.common.utilities.math.geometry.convex.shapes;

import fit.astro.vsa.common.utilities.math.geometry.convex.Point3D;
import fit.astro.vsa.common.utilities.math.geometry.convex.Polyhedron;
import fit.astro.vsa.common.utilities.math.geometry.convex.PolyhedronFace;
import fit.astro.vsa.common.utilities.math.geometry.convex.QuickHull3D;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;

/**
 * The generic polyhedron hexahedron object
 *
 * @author Kyle Johnston
 */
public class Hexahedron extends Polyhedron {

    /**
     * Generic Hexahedron Object
     *
     * @param mins The min vertex
     * @param maxs The max vertex
     * @param xSkew How to skew the cube in the x domain
     */
    public Hexahedron(Vector3D mins, Vector3D maxs, double xSkew) {
        //xSkew makes the top and bottom x different (so its not actually a cube)
        Point3D hhh = new Point3D(maxs.getX() + xSkew, maxs.getY(), maxs.getZ());
        Point3D hhl = new Point3D(maxs.getX() + xSkew, maxs.getY() + 2, mins.getZ());
        Point3D hlh = new Point3D(maxs.getX() - xSkew, mins.getY(), maxs.getZ());
        Point3D hll = new Point3D(maxs.getX() - xSkew, mins.getY(), mins.getZ());
        Point3D lhh = new Point3D(mins.getX() + xSkew, maxs.getY(), maxs.getZ());
        Point3D lhl = new Point3D(mins.getX() + xSkew, maxs.getY(), mins.getZ());
        Point3D llh = new Point3D(mins.getX() - xSkew, mins.getY(), maxs.getZ());
        Point3D lll = new Point3D(mins.getX() - xSkew, mins.getY(), mins.getZ());

        Point3D[] points = new Point3D[]{hhh, hhl, hlh, hll, lhh, lhl, llh, lll};

        QuickHull3D hull = new QuickHull3D();
        hull.build(points);

        Point3D[] vertices = hull.getVertices();
        int[][] faceIndices = hull.getHullFaces();

        // Convert Face to Polyhedron Face (really we should make them the same thing, or one extend the other....later)
        for (int i = 0; i < vertices.length; i++) {
            List<Point3D> pt3ds = new ArrayList<>();
            for (int k = 0; k < faceIndices[i].length; k++) {

                int idx = faceIndices[i][k];
                pt3ds.add(vertices[idx]);

            }

            faces.add(new PolyhedronFace(pt3ds));
        }

    }
}
