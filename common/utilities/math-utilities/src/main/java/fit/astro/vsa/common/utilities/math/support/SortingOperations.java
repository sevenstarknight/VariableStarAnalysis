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
package fit.astro.vsa.common.utilities.math.support;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class SortingOperations {

    // Empty Constructor
    private SortingOperations(){
        
    }
    /**
     * Ascending
     * <p>
     * @param <K>
     * @param <V>
     * @param map
     * <p>
     * @return
     */
    public static <K, V extends Comparable<? super V>> Map<K, V>
            sortByAcendingValue(Map<K, V> map) {
        List<Map.Entry<K, V>> list = new LinkedList<>(map.entrySet());

        Collections.sort(list, (Map.Entry<K, V> o1, Map.Entry<K, V> o2) 
                -> (o1.getValue()).compareTo(o2.getValue()));

        Map<K, V> result = new LinkedHashMap<>();
        list.stream().forEach((entry) -> {
            result.put(entry.getKey(), entry.getValue());
        });
        return result;
    }

    public static <K, V extends Comparable<? super V>> Map<K, V>
            sortByDecendingValue(Map<K, V> map) {
        List<Map.Entry<K, V>> list = new LinkedList<>(map.entrySet());

        Collections.sort(list, (Map.Entry<K, V> o1, Map.Entry<K, V> o2) 
                -> (o2.getValue()).compareTo(o1.getValue()));

        Map<K, V> result = new LinkedHashMap<>();
        list.stream().forEach((entry) -> {
            result.put(entry.getKey(), entry.getValue());
        });
        return result;
    }
}
