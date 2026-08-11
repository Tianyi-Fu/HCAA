% ===== SORTS START =====
#const n = 15.
sorts
#agent = {agent1}.
#user  = {user}.
#room  = {kitchen, living_room, bedroom, bathroom}.
#fruit = {apple, banana, orange, peach, grape}.
#vegetable = {tomato, carrot, potato, lettuce, mushroom}.
#baked = {bread, cake, cookie, pizza, muffin}.
#protein = {chicken, steak, salmon, egg, cheese}.
#drinkware = {cup, mug, glass, bottle, jar, kettle, teapot, thermos, beaker, coffeepot, cauldron, flask, carafe, pitcher}.
#fillable  = {cup, mug, glass, bottle, jar, kettle, teapot, thermos, bowl, bucket, vase, beaker, coffeepot, cauldron, flask, carafe, pitcher}.
#tableware = {plate, bowl, spoon, fork, knife}.
#servingware = {plate, bowl}.
#portable_media = {book, newspaper, phone, computer}.
#stationery = {pencil, crayon}.
#linen = {pillow, blanket, rug, curtain, towel}.
#decor = {vase, plant, candle, clock}.
#cleaning = {broom, mop, bucket, basket}.
#sit_furn = {chair, couch, bed, bench}.
#table_furn = {table, drawers}.
#appliance = {refrigerator, oven, stove, dishwasher, microwave_oven, hamper}.
#mounted_light = {lamp, chandelier}.
#mounted_media = {television}.
#food            = #fruit + #vegetable + #baked + #protein.
#kitchenware     = #drinkware + #tableware.
#large_household = #sit_furn + #table_furn + #appliance.
#household       = #linen + #decor + #cleaning.
#small_household = #portable_media + #stationery + #mounted_light + #mounted_media.
#heatable = #protein.
#item = #food + #kitchenware + #portable_media + #stationery + #household.
#on_furniture        = #sit_furn + #table_furn.
#surface             = #on_furniture + #servingware.
#inside_furniture    = #appliance.
#microwave_furniture = {microwave_oven}.
#light               = #mounted_light.
#switch_furniture    = #microwave_furniture + #light + #mounted_media + {stove} + {dishwasher}.
#container_furniture = #inside_furniture + #on_furniture.
#furniture           = #container_furniture + #switch_furniture.
#value = 0..10.
#sum_val = 0..100.
#step  = 0..n.
#thing = #item + #furniture.
#user_furniture = #furniture + #user.
% ===== SORTS END =====
% ----------------------------- fluents -----------------------------%
#inertial_fluent = location(#item, #room, #furniture) + location(#item, #user) + furniture_location(#furniture, #room) + user_location(#room) + locked(#inside_furniture) + open(#inside_furniture) + has(#agent, #item) + has(#user, #item) + in(#agent, #room) + changed(#inside_furniture) + at_furniture(#agent, #furniture) + at_user(#agent) + heated(#heatable) + inside(#inside_furniture, #item) + on(#surface, #item) + switched_on(#switch_furniture)+
switched_off(#switch_furniture)+ closed(#inside_furniture) + filled(#fillable, #fillable) .
#defined_fluent = dangerous(#agent).
#fluent = #inertial_fluent + #defined_fluent.
% ----------------------------- action ------------------------------%
#action = walk(#agent, #room) + walktowards(#agent, #user_furniture) + grab(#agent, #item, #container_furniture) + grab(#agent, #item, #user) + putin(#agent, #item, #inside_furniture) + put(#agent, #item, #on_furniture) + serveon(#agent, #food, #servingware) + give(#agent, #item, #user) + switchon(#agent, #switch_furniture) + switchoff(#agent, #switch_furniture) + open(#agent, #inside_furniture) + close(#agent, #inside_furniture) + fill(#agent, #fillable, #fillable).
% ----------------------------- predicates -----------------------------%
predicates
cost(#action,#value).
cost_defined(#action).
total(#sum_val).
inside(#furniture, #item).
on(#furniture, #item).
holds(#fluent, #step).
occurs(#action, #step).
success().
goal(#step).
something_happened(#step).
goal_1(#step).
goal_2(#step).
goal_rollback(#step).
goal_furniture_restored(#step).
at_furniture(#agent, #furniture).
show_last_holds(#fluent).
show_start_holds(#fluent).
show_changed_holds(#fluent).
show_changed_holds_name(#thing).
operated_thing(#thing).
show_operated_holds_name(#thing).
% -------------------------- rules --------------------------%
rules
show_start_holds(F) :- holds(F, 0).
show_last_holds(F)  :- holds(F, n).
show_changed_holds(F) :- holds(F, 0), not holds(F, n).
show_changed_holds(F) :- not holds(F, 0), holds(F, n).
operated_thing(Thing) :- occurs(grab(Agent, Thing, Furniture), Step), #item(Thing), #container_furniture(Furniture).
operated_thing(Thing) :- occurs(grab(Agent, Thing, User),      Step), #item(Thing), #user(User).
operated_thing(Thing) :- occurs(putin(Agent, Thing, Furniture), Step), #item(Thing), #inside_furniture(Furniture).
operated_thing(Thing) :- occurs(put(Agent, Thing, Furniture),   Step), #item(Thing), #on_furniture(Furniture).
operated_thing(Thing) :- occurs(serveon(Agent, Thing, Ware),    Step), #food(Thing), #servingware(Ware).
operated_thing(Thing) :- occurs(give(Agent, Thing, User),       Step), #item(Thing), #user(User).
operated_thing(Thing) :- occurs(switchon(Agent, Thing),         Step), #switch_furniture(Thing).
operated_thing(Thing) :- occurs(switchoff(Agent, Thing),        Step), #switch_furniture(Thing).
operated_thing(Thing) :- occurs(open(Agent, Thing),             Step), #inside_furniture(Thing).
operated_thing(Thing) :- occurs(close(Agent, Thing),            Step), #inside_furniture(Thing).
operated_thing(Thing) :- occurs(fill(Agent, Thing, Source),     Step), #fillable(Thing).
show_operated_holds_name(Thing) :- operated_thing(Thing).
show_changed_holds_name(T) :- #item(T),      not holds(location(T,R,F), 0), holds(location(T,R,F), n).
show_changed_holds_name(F) :- #furniture(F), not holds(location(T,R,F), 0), holds(location(T,R,F), n).
show_changed_holds_name(T) :- #item(T),              not holds(inside(F,T), 0), holds(inside(F,T), n).
show_changed_holds_name(F) :- #inside_furniture(F),  not holds(inside(F,T), 0), holds(inside(F,T), n).
show_changed_holds_name(T) :- #item(T),         not holds(on(F,T), 0), holds(on(F,T), n).
show_changed_holds_name(F) :- #on_furniture(F), not holds(on(F,T), 0), holds(on(F,T), n).
show_changed_holds_name(T) :- #item(T), not holds(has(Agent, T), 0), holds(has(Agent, T), n).
show_changed_holds_name(F) :- #inside_furniture(F), not holds(open(F),   0), holds(open(F),   n).
show_changed_holds_name(F) :- #inside_furniture(F), not holds(closed(F), 0), holds(closed(F), n).
show_changed_holds_name(F) :- #switch_furniture(F), not holds(switched_on(F),  0), holds(switched_on(F),  n).
show_changed_holds_name(F) :- #switch_furniture(F), not holds(switched_off(F), 0), holds(switched_off(F), n).
show_changed_holds_name(C) :- #fillable(C), #fillable(S), not holds(filled(C, S), 0), holds(filled(C, S), n).
%show_changed_holds_name(Thing) :- #thing(Thing), holds(locked(Thing), 0), -holds(locked(Thing), n),#inside_furniture(Thing).
%show_changed_holds_name(Thing) :- #thing(Thing), -holds(locked(Thing), 0), holds(locked(Thing), n),#inside_furniture(Thing).
%--------------------------------------------------------
% default cost for agent_actions is 0
%cost(A,0) :- #action(A), not cost_defined(A).
% cost for any actions
cost(walk(A, R), 2) :- occurs(walk(A, R), I).
cost(walktowards(A,U), 1) :- occurs(walktowards(A,U), I), #user(U).
cost(walktowards(A, F), 1) :- occurs(walktowards(A, F), I), #furniture(F).
cost(grab(A, T, F), 0) :- occurs(grab(A, T, F), I),#container_furniture(F).
cost(grab(A, T, U), 1) :- occurs(grab(A, T, U), I).
cost(putin(A, T, F), 1) :- occurs(putin(A, T, F), I),#inside_furniture(F).
cost(put(A, T, F), 0) :- occurs(put(A, T, F), I),#on_furniture(F).
cost(serveon(A, F, W), 0) :- occurs(serveon(A, F, W), I),#servingware(W).
cost(give(A, T, U), 1) :- occurs(give(A, T, U), I).
cost(switchon(A, F), 1) :- occurs(switchon(A, F), I),#switch_furniture(F).
cost(switchoff(A, F), 1) :- occurs(switchoff(A, F), I),#switch_furniture(F).
cost(open(A, F), 1) :- occurs(open(A, F), I),#inside_furniture(F).
cost(close(A, F), 1) :- occurs(close(A, F), I),#inside_furniture(F).
cost(fill(A, C, S), 1) :- occurs(fill(A, C, S), I),#fillable(C).
% whether give every action a cost
%cost_defined(A) :- cost(A,V), V != 0.
% calculate every actions cost
%total(Sum) :- Sum = #sum{V,I : cost(A,V), occurs(A,I)}.
%------------------------------------------------------------------
%walk from rooms
-holds(in(A, R1), I+1) :-
    occurs(walk(A, R2), I),
    holds(in(A, R1), I),
    R1 != R2.
holds(in(A, R2), I+1) :-
    occurs(walk(A, R2), I).
% Moving to a specific furniture inside the room
holds(at_furniture(A, F), I+1) :- occurs(walktowards(A, F), I), holds(in(A, R), I), holds(furniture_location(F, R), I), #furniture(F).
% Moving to the user's location inside the room
holds(at_user(A), I+1) :- occurs(walktowards(A, U), I), holds(in(A, R), I), holds(user_location(R), I), #user(U).
% Switching on furniture (e.g., microwave)
holds(switched_on(F), I+1) :- occurs(switchon(A, F), I), not holds(switched_on(F), I), holds(at_furniture(A, F), I),#switch_furniture(F).
-holds(switched_on(F), I+1) :- occurs(switchoff(A, F), I), holds(switched_on(F), I), holds(at_furniture(A, F), I),#switch_furniture(F).
-holds(switched_off(F), I+1) :- occurs(switchon(A, F), I), not holds(switched_on(F), I), holds(at_furniture(A, F), I),#switch_furniture(F).
holds(switched_off(F), I+1) :- occurs(switchoff(A, F), I), holds(switched_on(F), I), holds(at_furniture(A, F), I),#switch_furniture(F).
holds(filled(C, S), I+1) :- occurs(fill(A, C, S), I), not holds(filled(C, S), I), holds(has(A, C), I), holds(on(F, S), I), C != S, #fillable(C), #fillable(S), #surface(F).
% Running the dishwasher (switchon) washes its contents and returns them to the cupboard (drawers).
% Move the item's LOCATION (not just on/inside) so the dangling location(...,dishwasher)
% can't re-derive inside(dishwasher,T) at the next step (which would contradict the
% retraction below and make the run UNSAT).
holds(location(T, kitchen, drawers), I+1) :- occurs(switchon(A, dishwasher), I), holds(inside(dishwasher, T), I), holds(at_furniture(A, dishwasher), I).
-holds(location(T, R, dishwasher), I+1)   :- occurs(switchon(A, dishwasher), I), holds(location(T, R, dishwasher), I), holds(at_furniture(A, dishwasher), I).
-holds(inside(dishwasher, T), I+1)        :- occurs(switchon(A, dishwasher), I), holds(inside(dishwasher, T), I), holds(at_furniture(A, dishwasher), I).
% On/inside is true if the item is located in the furniture
holds(inside(F,T), I) :- holds(location(T,R,F), I), #inside_furniture(F).
holds(on(F,T), I)     :- holds(location(T,R,F), I), #on_furniture(F).
% Getting items from inside furniture
-holds(location(T, R, F), I+1) :- occurs(grab(A, T, F), I), holds(location(T, R, F), I), holds(in(A, R), I), holds(at_furniture(A, F), I), holds(open(F), I), holds(inside(F, T), I), #inside_furniture(F).
-holds(inside(F, T), I+1) :- occurs(grab(A, T, F), I), #inside_furniture(F).
holds(has(A, T), I+1) :- occurs(grab(A, T, F), I), #inside_furniture(F).
% Grabbing items from inside furniture
% Remove the item's location and inside relationship
-holds(location(T, R, F), I+1) :- occurs(grab(A, T, F), I),
    holds(location(T, R, F), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    holds(open(F), I),
    holds(inside(F, T), I),
    #inside_furniture(F).
-holds(inside(F, T), I+1) :- occurs(grab(A, T, F), I),
    holds(inside(F, T), I),
    holds(at_furniture(A, F), I),
    #inside_furniture(F).
% The agent now has the item
holds(has(A, T), I+1) :- occurs(grab(A, T, F), I),
    holds(location(T, R, F), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    holds(open(F), I),
    holds(inside(F, T), I),
    #inside_furniture(F).
% Grabbing items from on top of furniture
% Remove the item's location and on relationship
-holds(location(T, R, F), I+1) :- occurs(grab(A, T, F), I),
    holds(location(T, R, F), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    holds(on(F, T), I),
    #on_furniture(F).
-holds(on(F, T), I+1) :- occurs(grab(A, T, F), I),
    holds(on(F, T), I),
    holds(at_furniture(A, F), I),
    #on_furniture(F).
% The agent now has the item
holds(has(A, T), I+1) :- occurs(grab(A, T, F), I),
    holds(location(T, R, F), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    holds(on(F, T), I),
    #on_furniture(F).
-holds(location(T, user), I+1) :- occurs(grab(A, T, user), I),
    holds(location(T, user), I),
    holds(in(A, R), I),
    holds(user_location(R), I),
    holds(at_user(A), I).
-holds(has(user, T), I+1) :- occurs(grab(A, T, user), I),
    holds(has(user, T), I),
    holds(in(A, R), I),
    holds(user_location(R), I),
    holds(at_user(A), I).
% The agent now has the item
holds(has(A, T), I+1) :- occurs(grab(A, T, user), I),
    holds(location(T, user), I),
    holds(in(A, R), I),
    holds(user_location(R), I),
    holds(at_user(A), I).
% Placing items into furniture
holds(location(T, R, F), I+1) :- occurs(putin(A, T, F), I),
    holds(has(A, T), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    holds(open(F), I),
    #inside_furniture(F).
holds(inside(F, T), I+1) :- occurs(putin(A, T, F), I),
    holds(has(A, T), I),
    holds(at_furniture(A, F), I),
    #inside_furniture(F).
-holds(has(A, T), I+1) :- occurs(putin(A, T, F), I),
    holds(has(A, T), I),
    holds(at_furniture(A, F), I),
    #inside_furniture(F).
% Placing items onto furniture
holds(location(T, R, F), I+1) :- occurs(put(A, T, F), I),
    holds(has(A, T), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    #on_furniture(F).
holds(on(F, T), I+1) :- occurs(put(A, T, F), I),
    holds(has(A, T), I),
    holds(at_furniture(A, F), I),
    #on_furniture(F).
-holds(has(A, T), I+1) :- occurs(put(A, T, F), I),
    holds(has(A, T), I),
    holds(at_furniture(A, F), I),
    #on_furniture(F).
% ----- serveon: place food F onto servingware W (plate/bowl) the agent holds -----
holds(on(W, F), I+1) :- occurs(serveon(A, F, W), I),
    holds(has(A, F), I), holds(has(A, W), I),
    #food(F), #servingware(W).
-holds(has(A, F), I+1) :- occurs(serveon(A, F, W), I),
    holds(has(A, F), I), holds(has(A, W), I),
    #food(F), #servingware(W).
% Giving a loaded servingware delivers its contents to the user.
holds(on(W, F), I+1) :- holds(on(W, F), I), holds(has(U, W), I+1),
    #servingware(W), #food(F), #user(U).
% Placing items onto the user
% give(A,T,U) causes the robot to place item T onto the user U and no longer have T
holds(location(T,U),I+1):-occurs(give(A,T,U),I), holds(has(A,T),I),holds(in(A,R),I),holds(user_location(R),I),holds(at_user(A),I).
-holds(has(A,T),I+1):-occurs(give(A,T,U),I), holds(has(A,T),I),holds(in(A,R),I),holds(user_location(R),I),holds(at_user(A),I).
holds(has(U, T), I+1) :- occurs(give(A, T, U), I), holds(has(A, T), I), holds(in(A, R), I), holds(user_location(R), I), holds(at_user(A), I).
% If an item is located with the user, the user "has" the item
holds(has(user, T), I) :- holds(location(T, user), I).
% reverse: user holding an item implies it is located at the user (enables grab-back)
holds(location(T, user), I) :- holds(has(user, T), I), #item(T).
% If the item is no longer located with the user, the user no longer "has" the item
-holds(has(user, T), I+1) :- -holds(location(T, user), I+1), holds(has(user, T), I).
% Delivering heated beverage to the user's location clears the dangerous state
holds(location(T,user),I+1):-occurs(give(A,T,user),I), holds(has(A,T),I),holds(in(A,R),I),holds(user_location(R),I),holds(heated(T),I).
-holds(has(A,heated(T)),I+1):-occurs(give(A,T,user),I), holds(has(A,heated(T)),I),holds(in(A,R),I),holds(user_location(R),I).
% Operating furniture (open/close) requires the robot to be in front of the furniture
%holds(open(F), I+1):-occurs(open(A,F),I), not holds(open(F),I), holds(furniture_location(F,R),I), holds(in(A,R),I), holds(at_furniture(A,F),I), holds(changed(F),I+1),#inside_furniture(F).
%-holds(open(F), I+1):-occurs(close(A,F),I), holds(open(F),I), holds(furniture_location(F,R),I), holds(in(A,R),I), holds(at_furniture(A,F),I), holds(changed(F),I+1),#inside_furniture(F).
% Shelf needs the cellphone
%holds(open(bookshelf), I+1) :- occurs(open(A, bookshelf), I), not holds(open(bookshelf), I),holds(has(A, cellphone), I), holds(furniture_location(bookshelf, R), I), holds(in(A, R), I), holds(at_furniture(A, bookshelf), I), holds(changed(bookshelf), I+1).
%-holds(open(bookshelf), I+1) :- occurs(close(A, bookshelf), I), holds(open(bookshelf), I),holds(has(A, cellphone), I), holds(furniture_location(bookshelf, R), I), holds(in(A, R), I), holds(at_furniture(A, bookshelf), I), holds(changed(bookshelf), I+1).
% mark the changes of funitures
holds(changed(F),I+1):-occurs(open(A,F),I), not holds(changed(F),I),#inside_furniture(F).
-holds(changed(F),I+1):-occurs(open(A,F),I), holds(changed(F),I),#inside_furniture(F).
holds(changed(F),I+1):-occurs(close(A,F),I), not holds(changed(F),I),#inside_furniture(F).
-holds(changed(F),I+1):-occurs(close(A,F),I), holds(changed(F),I),#inside_furniture(F).
% Mark the changes to furniture
%holds(changed(F), I+1) :- occurs(switchon(A, F), I), not holds(changed(F), I).
%-holds(changed(F), I+1) :- occurs(switchon(A, F), I), holds(changed(F), I).
%holds(changed(F), I+1) :- occurs(switchoff(A, F), I), not holds(changed(F), I).
%-holds(changed(F), I+1) :- occurs(switchoff(A, F), I), holds(changed(F), I).
% When furniture is locked, it can't be used except open/close.
% For inside furniture
-occurs(grab(A, T, F), I) :- holds(inside(F, T), I), holds(locked(F), I), not holds(open(F), I), holds(furniture_location(F, R), I), holds(in(A, R), I), holds(at_furniture(A, F), I), #inside_furniture(F).
% For on furniture (though typically on_furniture are not locked, include for completeness)
%-occurs(grab(A, T, F), I) :- holds(on(F, T), I), holds(locked(F), I), holds(furniture_location(F, R), I), holds(in(A, R), I), holds(at_furniture(A, F), I), #on_furniture(F).
% needs a cellphone to open shelf
%-holds(locked(bookshelf), I+1) :- occurs(open(A, bookshelf), I), holds(locked(bookshelf), I), holds(has(A,cellphone), I), holds(furniture_location(bookshelf,R), I), holds(in(A,R), I), holds(at_furniture(A,bookshelf), I).
%holds(locked(bookshelf), I+1) :- occurs(close(A, bookshelf), I), not holds(locked(bookshelf), I), holds(has(A,cellphone), I), holds(furniture_location(bookshelf,R), I), holds(in(A,R), I), holds(at_furniture(A,bookshelf), I).
%change open/locked status of furnitures after open/close.
holds(open(F), I+1) :-
    occurs(open(A, F), I),
    not holds(open(F), I),
    holds(at_furniture(A,F), I),
    #inside_furniture(F).
-holds(closed(F), I+1) :-
    occurs(open(A, F), I),
    holds(closed(F), I),
    holds(at_furniture(A,F), I),
    #inside_furniture(F).
holds(closed(F), I+1) :-
    occurs(close(A,F), I),
    holds(open(F), I),
    holds(at_furniture(A,F), I),
    #inside_furniture(F).
-holds(open(F), I+1) :-
    occurs(close(A,F), I),
    holds(open(F), I),
    holds(at_furniture(A,F), I),
    #inside_furniture(F).% The item becomes heated when placed inside the microwave
holds(heated(T), I+1) :- occurs(switchon(A, F), I),
    holds(inside(F,T), I),
    holds(at_furniture(A, F), I),#microwave_furniture(F).
% ----------------------- state constraints -----------------------%
% Items in #hot_drink/#hot_food must be heated before they can be given to the user
%:- occurs(give(agent1, T, user), I), not holds(heated(T), I),#hot_food(T).
%:- occurs(give(agent1, T, user), I), not holds(heated(T), I),#hot_drink(T).
% The agent and user cannot both possess the same item at the same time
:- holds(has(agent1, T), I), holds(has(user, T), I).
% Entering dangerous state after obtaining any heated beverage
holds(dangerous(A),I):-holds(has(A,T),I),holds(heated(T),I).
% The robot cannot be in two places at once
-holds(in(A,R1),I):-holds(in(A,R2),I),R1!=R2.
-holds(in(A,R1),I+1):-occurs(walk(A,R2),I),R1!=R2.
-holds(at_furniture(A,F),I):- holds(at_user(A),I).
-holds(at_furniture(A,F),I+1):-occurs(walktowards(A,U),I),#user(U).
-holds(at_furniture(A,F),I+1):-occurs(walk(A,R2),I).
-holds(at_user(A),I+1):-occurs(walktowards(A,F),I),#furniture(F).
-holds(at_user(A),I+1):-occurs(walk(A,R2),I).
-holds(at_furniture(A,OldF),I+1):- occurs(walktowards(A,NewF),I), OldF != NewF,#furniture(OldF),#furniture(NewF).
% -------------------- executability conditions -------------------%
% Being in a dangerous state makes it impossible to perform actions other than walk, except for returning brewed_tea
-occurs(grab(A,T,F),I) :- holds(dangerous(A),I),#container_furniture(F).
-occurs(putin(A,T,F),I) :- holds(dangerous(A),I),#inside_furniture(F).
-occurs(put(A,T,F),I) :- holds(dangerous(A),I),#on_furniture(F).
-occurs(switchon(A, F), I) :- holds(dangerous(A), I),#switch_furniture(F).
% Robot can only give items on the user if it is in front of the user
-occurs(give(A,T,user),I):- not holds(at_user(A),I).
% need to near the container to open, and also close enough.
-occurs(grab(A,T,F),I):- not holds(in(A,R),I), holds(furniture_location(F,R),I), holds(location(T,R,F),I),#container_furniture(F).
%-occurs(grab(A,T,F),I):- holds(locked(F),I), holds(location(T,R,F),I),#container_furniture(F).
% The robot can only give an item if it has the item
-occurs(putin(A,T,F),I) :- not holds(has(A,T),I),#inside_furniture(F).
-occurs(put(A,T,F),I) :- not holds(has(A,T),I),#on_furniture(F).
% Robot can only open/close/switch furniture if it is in front of it
-occurs(open(A,F),I):- not holds(at_furniture(A,F),I),#inside_furniture(F).
-occurs(close(A,F),I):- not holds(at_furniture(A,F),I),#inside_furniture(F).
-occurs(switchon(A,F),I):- not holds(at_furniture(A,F),I),#switch_furniture(F).
-occurs(switchoff(A,F),I):- not holds(at_furniture(A,F),I),#switch_furniture(F).
% Robot can only grab items if it is in front of the furniture
-occurs(grab(A,T,F),I) :- not holds(at_furniture(A,F),I),#container_furniture(F).
% Robot can only give items if it is in front of the furniture
-occurs(putin(A,T,F),I) :- not holds(at_furniture(A,F),I),#inside_furniture(F).
-occurs(put(A,T,F),I) :- not holds(at_furniture(A,F),I),#on_furniture(F).
% The robot can only give an item in a furniture if it is in that room, and the furniture is either open or is a kitchen_table
-occurs(putin(A,T,F),I) :- not holds(in(A,R),I), holds(furniture_location(F,R),I),#inside_furniture(F).
-occurs(put(A,T,F),I) :- not holds(in(A,R),I), holds(furniture_location(F,R),I),#on_furniture(F).
% The robot can only grab an item if it is in the same room as the furniture, and the furniture is either open or is a kitchen_table
-occurs(grab(A,T,F),I):- not holds(in(A,R),I), holds(location(T,R,F),I),#container_furniture(F).
%-occurs(grab(A,T,F),I):- holds(locked(F),I), holds(location(T,R,F),I), F != kitchen_table, F != kitchen_counter,#container_furniture(F).
% Cannot grab an item if it's not inside or on the specified furniture
-occurs(grab(A, T, F), I) :- not holds(inside(F, T), I), not holds(on(F, T), I), holds(at_furniture(A, F), I),#container_furniture(F).
% The robot cannot walk to the same room it is currently in (no self-loops in movement)
:- occurs(walk(A,R1),I),holds(in(A,R1),I).
% Can't grab or give when the furniture is closed
% For inside furniture
-occurs(grab(A,T,F),I) :- not holds(open(F),I), #inside_furniture(F).
-occurs(putin(A,T,F),I) :- not holds(open(F),I), #inside_furniture(F).
% Can't grab an item if it's not in the specified furniture
-occurs(grab(A,T,F),I) :- not holds(location(T,R,F), I), holds(in(A,R), I).
%can't open/close/switch if they already be
:- occurs(open(A,F),I),holds(open(F),I),#inside_furniture(F).
:- occurs(close(A,F),I),-holds(open(F),I),#inside_furniture(F).
:- occurs(switchon(A,F),I),holds(switched_on(F),I),#switch_furniture(F).
:- occurs(switchoff(A,F),I),-holds(switched_on(F),I),#switch_furniture(F).
% Can only give items inside furniture that can contain items
-occurs(putin(A, T, F), I) :- not #inside_furniture(F).
% Furniture must be open to give items inside
%-occurs(putin(A, T, F), I) :- holds(locked(F), I), #inside_furniture(F).
% Can only give items on furniture that can have items placed on them
-occurs(put(A, T, F), I) :- not #on_furniture(F).
% Cannot use putin on furniture meant for items to be placed on
-occurs(putin(A, T, F), I) :- #on_furniture(F).
% Cannot use put on furniture meant for items to be placed inside
-occurs(put(A, T, F), I) :- #inside_furniture(F).
% The robot can only walk to a valid room
:- occurs(walk(A, Destination), I), not #room(Destination).
% The robot cannot walk towards furniture if already at it
:- occurs(walktowards(A, F), I), holds(at_furniture(A, F), I), #furniture(F).
% The robot cannot walk towards the user if already at the user
:- occurs(walktowards(A, user), I), holds(at_user(A), I).
% Cannot switch on microwave if there is no item inside
:- occurs(switchon(A, F), I), holds(open(F), I), #microwave_furniture(F).
holds(open(F), I+1) :- occurs(open(A, F), I), not holds(switched_on(F), I), #microwave_furniture(F).
%:- occurs(switchon(A, F), I), not holds(inside(F, T), I),#microwave_furniture(F).
% The microwave must be switched off before opening or closing
-occurs(open(A, F), I) :- holds(switched_on(F), I),#microwave_furniture(F).
-occurs(close(A, F), I) :- holds(switched_on(F), I),#microwave_furniture(F).
:- occurs(walk(A, R), I), holds(in(A, R), I).
-occurs(open(agent1, F), I) :- holds(has(agent1, T1), I), holds(has(agent1, T2), I), T1 != T2, #inside_furniture(F).
-occurs(switchon(agent1, F), I) :- holds(has(agent1, T1), I), holds(has(agent1, T2), I), T1 != T2,#switch_furniture(F).
-occurs(switchoff(agent1, F), I) :- holds(has(agent1, T1), I), holds(has(agent1, T2), I), T1 != T2,#switch_furniture(F).
-occurs(close(agent1, F), I) :- holds(has(agent1, T1), I), holds(has(agent1, T2), I), T1 != T2, #inside_furniture(F).
-occurs(grab(agent1, T, F), I) :- holds(has(agent1, T1), I), holds(has(agent1, T2), I), T1 != T2,#container_furniture(F).
:- holds(has(agent1, T1), I), holds(has(agent1, T2), I), holds(has(agent1, T3), I), T1 != T2, T1 != T3, T2 != T3.
-occurs(grab(A,T,F),I) :- holds(closed(F), I), #inside_furniture(F).
-occurs(putin(A,T,F),I) :- holds(closed(F), I), #inside_furniture(F).
:- holds(open(F), I),    holds(closed(F), I).
:- holds(switched_on(F), I), holds(switched_off(F), I).
%-occurs(grab(A,T,F),I):-holds(has(A,book),I),T!=book.
%-occurs(grab(A,book,F),I):-holds(has(A,T),I),T!=book.
% --------------------------- planning ---------------------------%
% Define goals
goal_furniture_restored(I) :- -holds(changed(microwave_oven), I).
% Define the overall goal
:- not success.
% Actions occur until the goal is achieved
occurs(A,I) | -occurs(A,I) :- not goal(I).
% Do not allow concurrent actions:
:- occurs(A1,I),occurs(A2,I),A1 != A2.
% An action occurs at each step before the goal is achieved:
something_happened(I) :- occurs(A,I).
:- goal(I), not goal(I-1),J < I,not something_happened(J).
% Minimize the number of movement steps
total(S) :- S = #sum{C, A:occurs(A,I), cost(A,C)}.
#minimize {V@2,V:total(V)}.
#minimize{1@1,I: occurs(A,I)}.
% Minimize the total number of actions (if desired)
%#minimize{1, I: occurs(A,I)}.
% -------------------- CWA for Defined Fluents -------------------%
-holds(F,I) :- #defined_fluent(F), not holds(F,I).
% --------------------- general Inertia Axiom --------------------%
holds(F,I+1) :- #inertial_fluent(F),holds(F,I),not -holds(F,I+1).
-holds(F,I+1) :- #inertial_fluent(F),-holds(F,I),not holds(F,I+1).
% ------------------------ CWA for Actions -----------------------%
-occurs(A,I) :- not occurs(A,I).
% ===== INITIAL CONDITIONS START =====
% ===== INITIAL CONDITIONS END =====
