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

import fit.astro.vsa.common.utilities.math.geometry.convex.shapes.Pyramid;
import fit.astro.vsa.common.utilities.math.geometry.convex.shapes.Tetrahedron;
import fit.astro.vsa.common.utilities.math.geometry.convex.shapes.Hexahedron;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLDouble;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.geometry.euclidean.threed.Vector3D;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Kyle Johnston
 */
public class ConvexHullAnalysis {
    
    private static final Logger LOGGER
            = LoggerFactory.getLogger(ConvexHullAnalysis.class);
    
    public static void main(String[] args) {
        
        List<MLArray> list = new ArrayList<>();
        
        StoreTetra(list);
        StoreHexahedron(list);
        StorePyramid(list);
        
        Path path = Paths.get("target", "science", "analysis",
                "IntersectionOfConvexHull.mat");
        try {
            Files.createDirectories(path);
            Files.deleteIfExists(path);
            Files.createFile(path);
            
            MatFileWriter writer = new MatFileWriter();
            writer.write(path.toFile(), list);
        } catch (IOException ex) {
            LOGGER.error("Unable to store to file", ex);
        }
        
    }
    
    private static void StoreTetra(List<MLArray> list) {
        Tetrahedron poly1 = new Tetrahedron(new Vector3D(-2, -2, -2),
                new Vector3D(2, 2, 2));
        Tetrahedron poly2 = new Tetrahedron(new Vector3D(-1, -5, -1),
                new Vector3D(1, 5, 1));
        
        Polyhedron update = poly2.clip(poly1);
        
        MLCell structureA = new MLCell("tetra_op_fov",
                new int[]{1, 2});
        
        structureA.cells().set(0, generateMLFromPoly(poly1, "poly1"));
        structureA.cells().set(1, generateMLFromPoly(poly2, "poly2"));
        
        list.add(structureA);
        
        MLCell structureB = new MLCell("tetra_fov",
                new int[]{1, 1});
        
        structureB.cells().set(0, generateMLFromPoly(update, "joint"));
        
        list.add(structureB);
    }
    
    private static void StorePyramid(List<MLArray> list) {
        Pyramid poly1 = new Pyramid(new Vector3D(-2, -2, -2),
                new Vector3D(2, 2, 2));
        Pyramid poly2 = new Pyramid(new Vector3D(-1, -5, -1),
                new Vector3D(1, 5, 1));
        
        Polyhedron update = poly1.clip(poly2);
        
        MLCell structureA = new MLCell("pyr_op_fov",
                new int[]{1, 2});
        
        structureA.cells().set(0, generateMLFromPoly(poly1, "poly1"));
        structureA.cells().set(1, generateMLFromPoly(poly2, "poly2"));
        
        list.add(structureA);
        
        MLCell structureB = new MLCell("pyr_fov",
                new int[]{1, 1});
        
        structureB.cells().set(0, generateMLFromPoly(update, "joint"));
        
        list.add(structureB);
    }
    
    private static void StoreHexahedron(List<MLArray> list) {
        Polyhedron poly1 = new Hexahedron(new Vector3D(-2, -2, -2),
                new Vector3D(0, 0, 0), 0.5);
        Polyhedron poly2 = new Hexahedron(new Vector3D(-1, -5, -1),
                new Vector3D(1, 5, 1), -2.5);
        
        Polyhedron update = poly1.clip(poly2);
        
        MLCell structureA = new MLCell("hex_op_fov",
                new int[]{1, 2});
        
        structureA.cells().set(0, generateMLFromPoly(poly1, "poly1"));
        structureA.cells().set(1, generateMLFromPoly(poly2, "poly2"));
        
        list.add(structureA);
        
        MLCell structureB = new MLCell("hex_fov",
                new int[]{1, 1});
        
        structureB.cells().set(0, generateMLFromPoly(update, "joint"));
        
        list.add(structureB);
    }
    
    private static MLDouble generateMLFromPoly(Polyhedron a, String label) {
        
        int counter = 0;
        double[][] ptsArray1 = new double[a.getVertices().size()][3];
        for (Point3D pts : a.getVertices()) {
            ptsArray1[counter] = pts.toArray();
            counter++;
        }
        
        return new MLDouble(label, ptsArray1);
        
    }
    
}
