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
package fit.astro.vsa.common.utilities.math.geometry.convex;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 * https://stackoverflow.com/questions/16389217/finding-the-intersection-of-two-3d-polygons
 * based on sudocode from
 * http://jhave.org/learner/misc/sutherlandhodgman/sutherlandhodgmanclipping.shtml
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu> (Edits and Updates)
 */
public class Polyhedron {

    protected List<PolyhedronFace> faces = new ArrayList<>();

    /**
     * Empty constructor
     */
    public Polyhedron() {
    }

    /**
     * Construct Polyhedron from Convex Hull
     *
     * @param vertices
     * @param faceIndices
     */
    public Polyhedron(Point3D[] vertices, int[][] faceIndices) {

        faces = new ArrayList<>();

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

    /**
     * The set of faces on the polygon (no order necessary
     *
     * @param faces
     */
    public Polyhedron(List<PolyhedronFace> faces) {
        this.faces = faces;
    }

    /**
     *
     * @return
     */
    public final Point3D getCenter() {

        Set<Point3D> vertices = getVertices();
        Point3D center = new Point3D(new double[]{0.0, 0.0, 0.0});
        Iterator<Point3D> iterVert = vertices.iterator();
        while (iterVert.hasNext()) {
            center = new Point3D(center.add(iterVert.next()));
        }

        center = new Point3D(
                center.scalarMultiply(1.0 / (double) vertices.size()));

        return center;
    }

    /**
     *
     * @param face
     */
    public final void addFace(PolyhedronFace face) {
        //for building face by face
        faces.add(face);
    }

    /**
     *
     * @param faceNumber
     * @return
     */
    public PolyhedronFace getFace(int faceNumber) {
        return faces.get(faceNumber);
    }

    /**
     * Get the Vertices of the Polyhedra
     *
     * @return
     */
    public Set<Point3D> getVertices() {

        List<Point3D> vertices = new ArrayList<>();
        faces.stream().forEach((face) -> {
            vertices.addAll(face.getVertices());
        });

        return new HashSet<>(vertices);
    }

    /**
     * Get the Faces of the Polyhedra
     *
     * @return
     */
    public List<PolyhedronFace> getFaces() {
        return faces;
    }

    // ========================================================
    /**
     *
     * @param clippingPolygon
     * @return
     */
    public Polyhedron clip(Polyhedron clippingPolygon) {
        Polyhedron workingPolygon = this;

        for (int i = 0; i < clippingPolygon.getFaces().size(); i++) {
            workingPolygon = clip(workingPolygon, clippingPolygon.getFace(i));
        }

        return workingPolygon;
    }

    private Polyhedron clip(Polyhedron inPolygon, PolyhedronFace clippingFace) {
        //each edges of each face of the inPolygon is clipped by the clipping face

        Polyhedron outPolygon = new Polyhedron();

        for (int i = 0; i < inPolygon.getFaces().size(); i++) {
            PolyhedronFace clippedFace = inPolygon.getFace(i).clipFace(clippingFace);
            if (clippedFace != null) {
                outPolygon.addFace(clippedFace);
            }
        }

        //additional step, clipping face is also clipped by the inPolygon
        //this step is what causes the requirement for both clipping and clipped 
        //shapes to be convex
        PolyhedronFace workingFace = clippingFace;
        for (int i = 0; i < inPolygon.getFaces().size(); i++) {
            if (workingFace == null) {
                //no need for bonus face in this case
                break;
            }
            workingFace = workingFace.clipFace(inPolygon.getFace(i));
        }
        if (workingFace != null) {
            outPolygon.addFace(workingFace);
        }

        return outPolygon;
    }
}
