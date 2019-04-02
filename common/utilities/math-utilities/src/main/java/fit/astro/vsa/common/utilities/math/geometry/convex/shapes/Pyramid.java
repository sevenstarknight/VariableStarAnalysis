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
 *
 * @author Kyle Johnston 
 */
public class Pyramid extends Polyhedron {

    /**
     * Generic Pyramid Object
     *
     * @param mins The min vertex
     * @param maxs The max vertex
     */
    public Pyramid(Vector3D mins, Vector3D maxs) {
        //xSkew makes the top and bottom x different (so its not actually a cube)
        Point3D hh = new Point3D(mins.getX(), mins.getY(), mins.getZ());
        Point3D hl = new Point3D(mins.getX(), maxs.getY(), mins.getZ());
        Point3D ll = new Point3D(maxs.getX(), mins.getY(), mins.getZ());
        Point3D lh = new Point3D(maxs.getX(), maxs.getY(), mins.getZ());

        Point3D peak = new Point3D((maxs.getX() - mins.getX()) / 2,
                (maxs.getY() - mins.getY()) / 2, maxs.getZ());

        Point3D[] points = new Point3D[]{hh, hl, ll, lh, peak};

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
