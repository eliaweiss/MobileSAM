from lineVision.LineCvUtils import LineCvUtils

from notebooks.Utils import calculate_iou


class ReduceBoxByIntersect:
    ################################################################    
    # find minimum intersection of all boxes
    def intersecting_box(boxes):
        # Initialize the smallest intersecting box with the first box
        smallest_box = boxes[0]

        # Iterate over the remaining boxes
        for box in boxes[1:]:
            # Check if the current box intersects with the smallest box
            if (box[0] <= smallest_box[0] + smallest_box[2] and
                box[0] + box[2] >= smallest_box[0] and
                box[1] <= smallest_box[1] + smallest_box[3] and
                box[1] + box[3] >= smallest_box[1]):
                # Update the smallest box if the current box is smaller
                if box[2] * box[3] < smallest_box[2] * smallest_box[3]:
                    smallest_box = box
            else:
                # If the current box does not intersect, return None
                return None

        return smallest_box
    
    ################################################################
    # find all box that intersects with box
    def find_intersect_boxes(box, boxes, threshold= 0.4):
        remaining_boxes = [boxes[j] for j in range(len(boxes)) if box != boxes[j]]
        if remaining_boxes:
            intersect_boxes = [box]
            for j, other_box in enumerate(remaining_boxes):
                intersect = LineCvUtils.calcBBIntersection(box, other_box)

                if intersect[0] >= threshold or intersect[1] >= threshold:
                    intersect_boxes.append(other_box)
        return intersect_boxes
    
    
    ################################################################    
    def find_box_with_max_iou(ibox, boxes):
        max_iou = 0
        box_i = -1
        for i, box in enumerate(boxes):
            iou = calculate_iou(box, ibox)
            if iou > max_iou:
                max_iou = iou
                box_i = i
        return box_i
    
    ################################################################    
    # main function    
    def reduceBoxByIntersect(boxes):
        selected_indices = []
        for box in boxes:
            intersect_boxes = ReduceBoxByIntersect.find_intersect_boxes(box, boxes)
            # print(intersect_boxes)
            # showImage_boxes(image, intersect_boxes)
            ibox = ReduceBoxByIntersect.intersecting_box(intersect_boxes)
            # print(ibox)
            box_i = ReduceBoxByIntersect.find_box_with_max_iou(ibox, boxes)
            # print(box_i)
            selected_indices.append(box_i)

        selected_indices = list(set(selected_indices))
        selected_box = []
        for i in selected_indices:
            selected_box.append(boxes[i])  
        return selected_box      
        