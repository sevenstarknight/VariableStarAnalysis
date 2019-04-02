/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without isEven the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.utilities.math.geometry.convex;

import fit.astro.vsa.common.utilities.math.NumericTests;
import java.util.ArrayList;
import java.util.List;

/**
 * NOTE: Face assumes that all its vertices are in the same place, this is not
 * checked and failure to conform to this will lead to errors. Face MUST have at
 * least 3 vertices by the time it is used, and the face itself must be convex.
 * Vertex winding must be anticlockwise, but a function rewind is available to
 * rewind if clockwise winding is used clockwise or anticlockwise winding must
 * be used, randomly putting in vertices will not end well
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu> (Edits and Updates)
 */
public class PolyhedronFace {

    List<Point3D> vertices = new ArrayList<>();

    /**
     * Empty Constructor
     */
    public PolyhedronFace() {
    }

    /**
     * Constructor with the set of CCW vertices
     *
     * @param vertices
     */
    public PolyhedronFace(List<Point3D> vertices) {
        this.vertices = vertices;
    }

    /**
     *
     * @return
     */
    public List<Point3D> getVertices() {
        return vertices;
    }

    /**
     *
     * @param vertex
     */
    public void addVertex(Point3D vertex) {
        if (Double.isNaN(vertex.getX())) {
            throw new RuntimeException("NaN Vertex");
        }

        if (Double.isInfinite(vertex.getX())
                || Double.isInfinite(vertex.getY())
                || Double.isInfinite(vertex.getZ())) {
            throw new RuntimeException("infinite Vertex");
        }

        if (vertices.contains(vertex)) {
            //degenerate vertex, do not add
        } else {
            vertices.add(vertex);
        }
    }

    /**
     * the winding of the vertices MUST be such that it looks anticlockwise from
     * the "outside", however, this method allows points to be added with either
     * clockwise or anticlockwise winding and then a final point that is
     * anywhere on the inside of the shape specified in this method and if the
     * wrong winding was used this rewinds it to anticlockwise winding
     *
     * @param internalPoint
     */
    public void rewind(Point3D internalPoint) {

        if (pointIsInsideFace(internalPoint) == false) {
            //winding is incorrect, reverese winding
            List<Point3D> verticesRewound = new ArrayList<>(vertices.size());
            for (int i = vertices.size() - 1; i >= 0; i--) {
                verticesRewound.add(vertices.get(i));
            }
            vertices = verticesRewound;
        }
    }

    /**
     * Number of edges on the face
     *
     * @return
     */
    public int getNumberOfEdges() {
        //(note because the last vertex connects to the first noOfEdges==noOfVerticies)
        return vertices.size(); 
    }

    /**
     * Get the vertex
     *
     * @param vertex
     * @return
     */
    public Point3D getVertex(int vertex) {
        return vertices.get(vertex);
    }

    /**
     * Get the start of the edge
     *
     * @param edgeNo
     * @return
     */
    public Point3D getStartOfEdge(int edgeNo) {
        return vertices.get(edgeNo);
    }

    /**
     *
     * @param edgeNo
     * @return
     */
    public Point3D getEndOfEdge(int edgeNo) {
        //%vertices.size() allows loop around for last edge
        return vertices.get((edgeNo + 1) % vertices.size()); 
    }

    private double getPointVsFaceDeterminant(Point3D point) {
       
        /**
         * the returned determinant is basically a measure of which side //(and
         * how far) a point lies from the plane
         * <p>
         * //FOR THIS TO WORK FACE MUST HAVE ANTICLOCKWISE WINDING WHEN LOOKED
         * AT //FROM OUTSIDE
         * <p>
         * we define faces as having their vertices in such an order that when
         * looked at from the outside the points are ordered anticlockswise SO
         * this function is equivalent to: pointIsInsideShape see
         * <p>
         * http://math.stackexchange.com/questions/214187/point-on-the-left-or-right-side-of-a-plane-in-3d-space
         * <p>
         * assuming face is convex, we only need the first 3 points to determine
         * the "winding" of the face
         */
        if (vertices.size() < 3) {
            throw new RuntimeException("Degenerate Face: Face has less than 3 vertices");
        }

        Point3D a = vertices.get(0);
        Point3D b = vertices.get(1);
        Point3D c = vertices.get(2);

        Point3D x = point;

        Point3D bDash = new Point3D(b.subtract(x));

        Point3D cDash = new Point3D(c.subtract(x));

        Point3D xDash = new Point3D(x.subtract(a));

        //find determinant of the 3 by 3 matrix described in link (see also: http://www.mathsisfun.com/algebra/matrix-determinant.html)
        double determinant = bDash.getX() * (cDash.getY() * xDash.getZ() - cDash.getZ() * xDash.getY()) - bDash.getY() * (cDash.getX() * xDash.getZ() - cDash.getZ() * xDash.getX()) + bDash.getZ() * (cDash.getX() * xDash.getY() - cDash.getY() * xDash.getX());

        return determinant;
    }

    /**
     *
     * @param point
     * @return
     */
    public boolean pointIsInsideFace(Point3D point) {
        /**
         * we define faces as having their vertices in such an order that when
         * looked at from the outside the points are ordered anticlockswise SO
         * this function is equivalent to: pointIsInsideShape see
         * http://math.stackexchange.com/questions/214187/point-on-the-left-or-right-side-of-a-plane-in-3d-space
         * find determinant of the 3 by 3 matrix described in link (see also:
         * http://www.mathsisfun.com/algebra/matrix-determinant.html)
         */
        double determinant = getPointVsFaceDeterminant(point);
        // <= because we define on the face to be "inside the face"
        return determinant <= 0;
    }

    /**
     * http://mathworld.wolfram.com/Plane.html
     * 
     * @param rayPoint1
     * @param rayPoint2
     * @return
     */
    public Point3D getIntersectionPoint(Point3D rayPoint1, Point3D rayPoint2) {
        //NOTE: This method treats the face as if it was an infinite plane
        //this treating as a plane is why convex shapes must be used
        //changed from above method as that can get upset with parallel lines
        double determinantPoint1 = getPointVsFaceDeterminant(rayPoint1);
        double determinantPoint2 = getPointVsFaceDeterminant(rayPoint2);

        if (NumericTests.isApproxEqual(determinantPoint1, determinantPoint2)) {
            /**
             * parallel line, if we've got into this method then it'll probably
             * be in the plane, the line is in the plane, the middle seems the
             * most reasonable point
             */

            Point3D average = new Point3D(rayPoint1.add(rayPoint2));
            average = new Point3D(average.scalarMultiply(0.5));

            return average;
        } else {
            /**
             * we want to return the point where the determinant would have been
             * zero
             * <p>
             * linear interpolation
             */
            Point3D intersect = new Point3D(rayPoint2.subtract(rayPoint1));
            intersect = new Point3D(intersect.scalarMultiply(
                    (0 - determinantPoint1)
                    / (determinantPoint2
                    - determinantPoint1)));

            intersect = new Point3D(intersect.add(rayPoint1));

            return intersect;
        }
    }

    /**
     * https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
     *
     * @param clippingFace
     * @return
     */
    public PolyhedronFace clipFace(PolyhedronFace clippingFace) {

        /**
         * Note, this face may be entirely clipped by the clipping face or
         * clipped to a degenerate edge, in this case null is returned
         */
        PolyhedronFace workingFace = new PolyhedronFace();

        for (int i = 0; i < this.getNumberOfEdges(); i++) {
            /**
             * clips all the edges of the working polygon against a plane based
             * upon the clipping face for each edge there are 4 cases, we must
             * determine which it is where we refer to starting and ending
             * vertices they are of workingFace where we refer to "the Face"
             * that is the clipping face and endEdge. The edge of the clipping
             * polygon
             * <p>
             * case 1) both starting vertices are inside face case 2) starting
             * vertex is inside face, ending vertex is inside case 3) Both
             * vertices are outside the face case 4) starting is outside the
             * face, ending is inside
             */
            Point3D point1 = getStartOfEdge(i);
            Point3D point2 = getEndOfEdge(i);

            if (clippingFace.pointIsInsideFace(point1) && clippingFace.pointIsInsideFace(point2)) {
                //case 1, the end point is added
                workingFace.addVertex(point2);
            } else if (clippingFace.pointIsInsideFace(point1) && clippingFace.pointIsInsideFace(point2) == false) {
                //case 2, only the intersection is added
                Point3D intersection = clippingFace.getIntersectionPoint(point1, point2);
                workingFace.addVertex(intersection);
            } else if (clippingFace.pointIsInsideFace(point1) == false && clippingFace.pointIsInsideFace(point2) == false) {
                //case 3, both vertices are outside the clip shape line, no vertexes added
            } else {
                //case 4 the ending vertex is inside and the starting vertex is outside
                //the line
                //the intercept and the end point are added
                Point3D intersection = clippingFace.getIntersectionPoint(point1, point2);

                workingFace.addVertex(intersection);
                workingFace.addVertex(point2);
            }
        }

        if (workingFace.getNumberOfEdges() >= 3) {
            return workingFace;
        } else {
            //degenerate or completely culled face
            return null;
        }

    }
}
