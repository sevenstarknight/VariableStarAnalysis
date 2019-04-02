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
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;

/**
 *
 * @author Kyle Johnston 
 */
public class Tetrahedron extends Polyhedron {

    /**
     * Generic Pyramid Object
     *
     * @param mins The min vertex
     * @param maxs The max vertex
     */
    public Tetrahedron(Vector3D mins, Vector3D maxs) {
        //xSkew makes the top and bottom x different (so its not actually a cube)
        Point3D hh = new Point3D(mins.getX(), mins.getY(), mins.getZ());
        Point3D hl = new Point3D(mins.getX(), maxs.getY(), mins.getZ());
        Point3D ll = new Point3D(maxs.getX(), mins.getY(), mins.getZ());

        Point3D peak = new Point3D((maxs.getX() + mins.getX()) / 2,
                (maxs.getY() + mins.getY()) / 2, maxs.getZ());

        Point3D center = new Point3D(hh);
        center = new Point3D(center.add(hh));
        center = new Point3D(center.add(hl));
        center = new Point3D(center.add(ll));
        center = new Point3D(center.scalarMultiply(1.0 / 4.0));

        PolyhedronFace top = new PolyhedronFace();
        PolyhedronFace bottom = new PolyhedronFace();
        PolyhedronFace north = new PolyhedronFace();
        PolyhedronFace east = new PolyhedronFace();

        north.addVertex(hh);
        north.addVertex(hl);
        north.addVertex(peak);
        north.rewind(center);

        top.addVertex(hl);
        top.addVertex(ll);
        top.addVertex(peak);
        top.rewind(center);

        east.addVertex(hh);
        east.addVertex(ll);
        east.addVertex(peak);
        east.rewind(center);

        bottom.addVertex(hh);
        bottom.addVertex(hl);
        bottom.addVertex(ll);
        bottom.rewind(center);

        addFace(top);
        addFace(bottom);
        addFace(north);
        addFace(east);
    }

}
